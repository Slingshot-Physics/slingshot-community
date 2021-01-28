#ifndef GJK_TESTS_HEADER
#define GJK_TESTS_HEADER

#include "epa_types.hpp"
#include "geometry_type_converters.hpp"
#include "mesh.hpp"
#include "quickhull.hpp"
#include "triangle.hpp"

namespace test_utils
{
   // Expects vertices of each mesh to be in world coordinates.
   // Returns 1 if successful. Returns -1 if unsuccessful.
   int generateMdHull(
      const geometry::types::triangleMesh_t & mesh_a,
      const geometry::types::triangleMesh_t & mesh_b,
      geometry::types::triangleMesh_t & md_hull
   )
   {
      geometry::types::minkowskiDiffVertex_t * minkowski_diff_polyhedron = \
         new geometry::types::minkowskiDiffVertex_t[mesh_a.numVerts * mesh_b.numVerts];
      unsigned int num_md_verts = 0;

      for (int i = 0; i < mesh_a.numVerts; ++i)
      {
         Vector3 vert_a = mesh_a.verts[i];
         for (int j = 0; j < mesh_b.numVerts; ++j)
         {
            Vector3 vert_b = mesh_b.verts[j];
            minkowski_diff_polyhedron[num_md_verts].bodyAVertId = i;
            minkowski_diff_polyhedron[num_md_verts].bodyBVertId = j;
            minkowski_diff_polyhedron[num_md_verts].vert = vert_a - vert_b;
            ++num_md_verts;
         }
      }

      // Generate the convex hull out of the MD vertices
      int result = geometry::mesh::generateHull(
         num_md_verts, minkowski_diff_polyhedron, md_hull
      );

      delete [] minkowski_diff_polyhedron;
      return result;
   }

   // Returns true if there is an intersection, false otherwise.
   bool intersection_test(
      const geometry::types::triangleMesh_t & mesh_a,
      const geometry::types::triangleMesh_t & mesh_b
   )
   {
      geometry::types::triangleMesh_t md_hull;
      int result = generateMdHull(mesh_a, mesh_b, md_hull);
      Vector3 zero;
      return geometry::mesh::pointInsideMesh(md_hull, zero);
   }

   bool intersection_test(
      const geometry::types::triangleMesh_t & mesh_a,
      const geometry::types::triangleMesh_t & mesh_b,
      float & closest_points_distance
   )
   {
      geometry::types::triangleMesh_t md_hull;
      int result = generateMdHull(mesh_a, mesh_b, md_hull);
      Vector3 zero;
      return geometry::mesh::pointInsideMesh(
         md_hull, zero, closest_points_distance
      );
   }

   // None of the body-specific information is available because Quickhull doesn't
   // return a mesh configuration with vertex labels.
   geometry::types::epaResult_t epa_test(
      const geometry::types::triangleMesh_t & md_hull
   )
   {
      geometry::types::epaResult_t epa_result;
      epa_result.collided = true;

      geometry::types::pointBaryCoord_t closest_pt;
      float closest_dist = 1e8f;
      int closest_triangle_id = -1;

      Vector3 zero;
      for (int i = 0; i < md_hull.numTriangles; ++i)
      {
         const geometry::types::meshTriangle_t & temp_mesh_triangle = \
            md_hull.triangles[i];
         geometry::types::triangle_t temp_triangle = {
            md_hull.verts[temp_mesh_triangle.vertIds[0]],
            md_hull.verts[temp_mesh_triangle.vertIds[1]],
            md_hull.verts[temp_mesh_triangle.vertIds[2]]
         };
         geometry::types::pointBaryCoord_t temp_closest_pt = \
            geometry::triangle::closestPointToPoint(
               temp_triangle.verts[0],
               temp_triangle.verts[1],
               temp_triangle.verts[2],
               zero
            );

         float temp_dist = temp_closest_pt.point.magnitude();
         if (temp_dist < closest_dist)
         {
            closest_dist = temp_dist;
            closest_pt = temp_closest_pt;
            closest_triangle_id = i;
         }
      }

      unsigned int vertId0 = md_hull.triangles[closest_triangle_id].vertIds[0];
      unsigned int vertId1 = md_hull.triangles[closest_triangle_id].vertIds[1];
      unsigned int vertId2 = md_hull.triangles[closest_triangle_id].vertIds[2];

      epa_result.p = closest_pt.point;

      return epa_result;
   }
}

#endif
