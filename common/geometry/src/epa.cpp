#include "epa.hpp"

#include "attitudeutils.hpp"
#include "geometry.hpp"
#include "gjk.hpp"
#include "mesh_ops.hpp"
#include "support_functions.hpp"
#include "transform.hpp"

namespace geometry
{

namespace epa
{
   // Finds the closest triangle on the MD mesh to the origin by finding the
   // triangle with the smallest min-norm point.
   unsigned int findClosestTriangleId(
      const geometry::types::epa::epaMesh_t & mesh,
      const Vector3 & interior_point,
      geometry::types::pointBaryCoord_t & closest_point,
      Vector3 & closest_tri_normal
   )
   {
      unsigned int closest_tri_id = -1;
      closest_point.point.Initialize(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);
      Vector3 zero(0.0f, 0.0f, 0.0f);

      const geometry::types::epa::EpaTriangles & triangles = mesh.triangles;
      for (unsigned int i = 0; i < triangles.size(); ++i)
      {
         geometry::types::triangle_t triangle = {
            {
               mesh.mdVerts[triangles[i].vertIds[0]].vert,
               mesh.mdVerts[triangles[i].vertIds[1]].vert,
               mesh.mdVerts[triangles[i].vertIds[2]].vert
            }
         };

         geometry::types::pointBaryCoord_t temp_point = \
            geometry::triangle::closestPointToPoint(
               triangle.verts[0], triangle.verts[1], triangle.verts[2], zero
            );

         if (
            (temp_point.point.magnitudeSquared() <= closest_point.point.magnitudeSquared())
         )
         {
            closest_tri_id = i;
            closest_point = temp_point;
            closest_tri_normal = calcNormal(triangle.verts, interior_point);
         }
      }

      return closest_tri_id;
   }

   // Delete the triangles in the mesh that have the support point in their
   // positive half-spaces.
   // Start at the end of the triangles in the mesh and go to the front. This lets
   // you delete triangles in-place without complicated logic to handle the loop
   // variable.
   int deleteVisibleTriangles(
      geometry::types::epa::epaMesh_t & mesh,
      const Vector3 & interior_point,
      const Vector3 & epa_support_point
   )
   {
      int num_deleted_triangles = 0;
      unsigned int j = mesh.triangles.size();
      while(j != 0)
      {
         Vector3 normal = calcNormal(
            mesh.mdVerts[mesh.triangles[j - 1].vertIds[0]].vert,
            mesh.mdVerts[mesh.triangles[j - 1].vertIds[1]].vert,
            mesh.mdVerts[mesh.triangles[j - 1].vertIds[2]].vert,
            interior_point
         );

         Vector3 tri_center = (
            mesh.mdVerts[mesh.triangles[j - 1].vertIds[0]].vert +
            mesh.mdVerts[mesh.triangles[j - 1].vertIds[1]].vert +
            mesh.mdVerts[mesh.triangles[j - 1].vertIds[2]].vert
         ) / 3.0f;

         geometry::types::pointBaryCoord_t closest_point = \
            geometry::triangle::closestPointToPoint(
               mesh.mdVerts[mesh.triangles[j - 1].vertIds[0]].vert,
               mesh.mdVerts[mesh.triangles[j - 1].vertIds[1]].vert,
               mesh.mdVerts[mesh.triangles[j - 1].vertIds[2]].vert,
               epa_support_point
            );

         // Delete the triangle if the support point dotted with the face's
         // normal is > 0.0f (the face can see the support point) and the
         // closest point on the MD mesh triangle is greater than a threshold
         // distance.
         if (
            (normal.dot(epa_support_point - tri_center) > 0.0f) &&
            ((closest_point.point - epa_support_point).magnitude() > 1e-5f)
         )
         {
            mesh.triangles.remove(j - 1);
            ++num_deleted_triangles;
         }
         --j;
      }

      return num_deleted_triangles;
   }

   // Find the pairs of vertex IDs on the deleted triangles that share edges
   // with the remaining triangles by looping over the remaining triangles and
   // finding neighbor IDs of -1. Adds new triangles to the mesh using the
   // abandoned neighbor IDs.
   int addNewTriangles(
      geometry::types::epa::epaMesh_t & mesh, int new_vert_index
   )
   {
      unsigned int num_start_tris = mesh.triangles.size();
      for (unsigned int i = 0; i < num_start_tris; ++i)
      {
         for (unsigned int j = 0; j < 3; ++j)
         {
            // Generate a new triangle for each pair with the support point
            // index as the third vertex ID.
            if (mesh.triangles[i].neighborIds[j] == -1)
            {
               unsigned int vert0 = geometry::types::epa::EpaTriangles::edgeVerts[j][0];
               unsigned int vert1 = geometry::types::epa::EpaTriangles::edgeVerts[j][1];

               int result = mesh.triangles.add_triangle(
                  mesh.triangles[i].vertIds[vert0],
                  mesh.triangles[i].vertIds[vert1],
                  new_vert_index
               );

               if (result < 0)
               {
                  return -1;
               }
            }
         }
      }

      return 1;
   }
}

}
