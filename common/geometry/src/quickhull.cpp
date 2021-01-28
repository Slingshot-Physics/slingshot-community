#include "quickhull.hpp"

#include "line.hpp"
#include "mesh_ops.hpp"
#include "triangle.hpp"
#include "triangle_graph_mesh.hpp"

#include <cassert>

namespace geometry
{

namespace mesh
{

   int quickhullMesh_t::addVert(const md_vert_t & md_vert)
   {
      if (numVerts >= MAX_VERTICES)
      {
         return -1;
      }

      verts[numVerts] = md_vert;

      ++numVerts;

      return numVerts - 1;
   }

   void convertQuickHullToMeshConfig(
      const quickhullMesh_t & qhull,
      const Vector3 & hull_center,
      geometry::types::triangleMesh_t & convex_hull
   )
   {
      // Have to set this by looping, some verts might be dead.
      convex_hull.numVerts = qhull.numVerts;

      for (unsigned int i = 0; i < qhull.numVerts; ++i)
      {
         convex_hull.verts[i] = qhull.verts[i].vert;
      }

      convex_hull.numTriangles = qhull.triangles.size();
      for (unsigned int i = 0; i < qhull.triangles.size(); ++i)
      {
         geometry::types::triangle_t temp_triangle;
         for (unsigned int j = 0; j < 3; ++j)
         {
            temp_triangle.verts[j] = qhull.verts[qhull.triangles[i].vertIds[j]].vert;
            convex_hull.triangles[i].vertIds[j] = qhull.triangles[i].vertIds[j];
         }

         convex_hull.triangles[i].normal = calcNormal(
            temp_triangle.verts, hull_center
         );
         convex_hull.triangles[i].normal.Normalize();
      }
   }

   void findTetrahedron(
      unsigned int num_points,
      const md_vert_t * points,
      quickhullMesh_t & qhull,
      bool * verts_added
   )
   {
      qhull.verts[qhull.numVerts] = points[0];
      ++qhull.numVerts;
      verts_added[0] = true;

      {
         int farthest_index = 1;
         md_vert_t farthest_point = points[1];
         float max_dist = (farthest_point.vert - qhull.verts[0].vert).magnitude();
         for (unsigned int i = 2; i < num_points; ++i)
         {
            float temp_dist = (points[i].vert - qhull.verts[0].vert).magnitude();
            if (temp_dist > max_dist)
            {
               max_dist = temp_dist;
               farthest_point = points[i];
               farthest_index = i;
            }
         }

         qhull.verts[qhull.numVerts] = farthest_point;
         ++qhull.numVerts;
         verts_added[farthest_index] = true;
      }

      {
         int farthest_index = 0;
         md_vert_t farthest_point = qhull.verts[0];
         float max_dist = 0.f;
         for (unsigned int i = 1; i < num_points; ++i)
         {
            Vector3 closest_point_on_line = geometry::line::closestPointToPoint(
               qhull.verts[0].vert, qhull.verts[1].vert, points[i].vert
            );
            float temp_dist = (closest_point_on_line - points[i].vert).magnitude();

            if (temp_dist > max_dist)
            {
               max_dist = temp_dist;
               farthest_point = points[i];
               farthest_index = i;
            }
         }

         qhull.verts[qhull.numVerts] = farthest_point;
         ++qhull.numVerts;
         verts_added[farthest_index] = true;
      }

      {
         int farthest_index = 0;
         Vector3 normal = (
            (qhull.verts[0].vert - qhull.verts[1].vert).crossProduct(
               qhull.verts[2].vert - qhull.verts[1].vert
            )
         );

         md_vert_t farthest_point = qhull.verts[0];
         float max_dist = 0.f;
         {
            for (unsigned int i = 1; i < num_points; ++i)
            {
               float temp_dist = fabs(
                  normal.dot(points[i].vert - qhull.verts[0].vert)
               );
               if (temp_dist > max_dist)
               {
                  max_dist = temp_dist;
                  farthest_point = points[i];
                  farthest_index = i;
               }
            }
         }

         qhull.verts[qhull.numVerts] = farthest_point;
         ++qhull.numVerts;
         verts_added[farthest_index] = true;
      }

      qhull.triangles.add_triangle(0, 1, 2);
      qhull.triangles.add_triangle(0, 1, 3);
      qhull.triangles.add_triangle(0, 2, 3);
      qhull.triangles.add_triangle(1, 2, 3);
   }

   int findFarthestPoint(
      const quickhullMesh_t & qhull,
      const Vector3 & hull_center,
      const bool * verts_added,
      unsigned int num_points,
      const md_vert_t * points
   )
   {
      int farthest_index = -1;
      float max_dist = 0.f;

      for (unsigned int i = 0; i < num_points; ++i)
      {
         for (unsigned int j = 0; j < qhull.triangles.size(); ++j)
         {
            Vector3 temp_verts[3] = {
               qhull.verts[qhull.triangles[j].vertIds[0]].vert,
               qhull.verts[qhull.triangles[j].vertIds[1]].vert,
               qhull.verts[qhull.triangles[j].vertIds[2]].vert
            };
            Vector3 normal = calcNormal(temp_verts, hull_center);

            if (
               ((points[i].vert - temp_verts[0]).dot(normal) > 0.f) &&
               !verts_added[i]
            )
            {
               float temp_dist = (temp_verts[0] - points[i].vert).magnitude();

               if (temp_dist > max_dist)
               {
                  max_dist = temp_dist;
                  farthest_index = i;
               }
            }
         }
      }

      return farthest_index;
   }

   void deleteTrianglesFacingPoint(
      const md_vert_t & point, const Vector3 & hull_center, quickhullMesh_t & qhull
   )
   {
      for (int i = qhull.triangles.size() - 1; i > -1; --i)
      {
         Vector3 temp_verts[3] = {
            qhull.verts[qhull.triangles[i].vertIds[0]].vert,
            qhull.verts[qhull.triangles[i].vertIds[1]].vert,
            qhull.verts[qhull.triangles[i].vertIds[2]].vert
         };
         Vector3 normal = calcNormal(temp_verts, hull_center);

         if (
            normal.dot(point.vert - temp_verts[0]) >= 0.f ||
            normal.dot(point.vert - temp_verts[1]) >= 0.f ||
            normal.dot(point.vert - temp_verts[2]) >= 0.f
         )
         {
            qhull.triangles.remove(i);
         }
      }
   }

   int addTrianglesToVertId(int new_vert_id, quickhullMesh_t & qhull)
   {
      if (new_vert_id < 0)
      {
         return 0;
      }
      unsigned int original_size = qhull.triangles.size();
      for (unsigned int i = 0; i < original_size; ++i)
      {
         for (unsigned int j = 0; j < 3; ++j)
         {
            if (qhull.triangles[i].neighborIds[j] == -1)
            {
               unsigned int vert0 = geometry::mesh::TriangleGraphBase::edgeVerts[j][0];
               unsigned int vert1 = geometry::mesh::TriangleGraphBase::edgeVerts[j][1];

               qhull.triangles.add_triangle(
                  qhull.triangles[i].vertIds[vert0],
                  qhull.triangles[i].vertIds[vert1],
                  new_vert_id
               );
            }
         }
      }

      return 1;
   }

   int generateHull(
      unsigned int num_points,
      const md_vert_t * points,
      geometry::types::triangleMesh_t & convex_hull
   )
   {
      convex_hull.numTriangles = 0;
      convex_hull.numVerts = 0;
      int result = 1;
      assert(num_points >= 4);

      bool * verts_added = new bool[num_points];
      std::memset(verts_added, 0, num_points * sizeof(bool));

      quickhullMesh_t quick_hull = {{}, 0, {}};
      
      findTetrahedron(num_points, points, quick_hull, verts_added);

      Vector3 hull_center;
      for (unsigned int i = 0; i < quick_hull.numVerts; ++i)
      {
         hull_center += quick_hull.verts[i].vert / (float )(quick_hull.numVerts);
      }

      if (num_points == 4)
      {
         convertQuickHullToMeshConfig(quick_hull, hull_center, convex_hull);
         delete[] verts_added;
         return 1;
      }

      int num_iters = 0;
      while (quick_hull.numVerts < MAX_VERTICES)
      {
         // Loop over all vertices, find the index of one vertex that's
         // farthest from any existing triangle in the mesh and exterior to the
         // mesh. If this returns -1, then there is no point 'outside' of the
         // mesh that can be added. E.g. the hull is complete.
         int farthest_point_index = findFarthestPoint(
            quick_hull,
            hull_center,
            verts_added,
            num_points,
            points
         );
         if (farthest_point_index < 0)
         {
            break;
         }
         md_vert_t farthest_point = points[farthest_point_index];

         // Delete the triangles in the mesh that face the farthest point.
         deleteTrianglesFacingPoint(farthest_point, hull_center, quick_hull);

         // Add the farthest point to the set of vertices.
         int qhull_vert_index = quick_hull.addVert(farthest_point);
         if (qhull_vert_index < 0)
         {
            delete[] verts_added;
            return -1;
         }
         verts_added[farthest_point_index] = true;

         // Redraw triangles from the hole to the new vertex.
         int result = addTrianglesToVertId(qhull_vert_index, quick_hull);
         if (result < 0)
         {
            delete[] verts_added;
            return -1;
         }
         ++num_iters;
      }

      delete[] verts_added;

      for (unsigned int i = 0; i < quick_hull.triangles.size(); ++i)
      {
         assert(quick_hull.triangles[i].neighborIds[0] != -1);
         assert(quick_hull.triangles[i].neighborIds[1] != -1);
         assert(quick_hull.triangles[i].neighborIds[2] != -1);

         assert(quick_hull.triangles[i].vertIds[0] != -1);
         assert(quick_hull.triangles[i].vertIds[1] != -1);
         assert(quick_hull.triangles[i].vertIds[2] != -1);
      }

      convertQuickHullToMeshConfig(quick_hull, hull_center, convex_hull);
      return result;
   }

   int generateHull(
      unsigned int num_points,
      const Vector3 * points,
      geometry::types::triangleMesh_t & mesh_out
   )
   {
      mesh_out.numVerts = 0;
      mesh_out.numTriangles = 0;
      md_vert_t * temp_points = new md_vert_t[num_points];
      for (unsigned int i = 0; i < num_points; ++i)
      {
         temp_points[i].bodyAVertId = -1;
         temp_points[i].bodyBVertId = -1;
         temp_points[i].vert = points[i];
      }

      int result = generateHull(num_points, temp_points, mesh_out);

      delete[] temp_points;

      return result;
   }

}

}
