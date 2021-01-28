#include "triangle.cuh"

#include "plane.cuh"
#include "segment.cuh"

namespace cugeom
{

namespace triangle
{

   __constant__ edge_t triangle_edge_combos[3] = {
      {0, 1, 2},
      {1, 2, 0},
      {2, 0, 1}
   };

   CUDA_CALLABLE cumath::Vector3 baryCoords(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & q
   )
   {
      cumath::Vector3 v0 = b - a;
      cumath::Vector3 v1 = c - a;
      cumath::Vector3 v2 = q - a;

      float v0m2 = v0.magnitudeSquared();
      float v1m2 = v1.magnitudeSquared();
      float v0dv1 = v0.dot(v1);
      float v0dv2 = v0.dot(v2);
      float v1dv2 = v1.dot(v2);

      float det = v0m2 * v1m2 - v0dv1 * v0dv1;

      // cumath::Vector3 bary;
      // bary[1] = (v1m2 * v0dv2 - v0dv1 * v1dv2)/det;
      // bary[2] = (v0m2 * v1dv2 - v0dv1 * v0dv2)/det;
      // bary[0] = 1.0f - bary[1] - bary[2];

      cumath::Vector3 bary(
         1.0f - bary[1] - bary[2],
         (v1m2 * v0dv2 - v0dv1 * v1dv2)/det,
         (v0m2 * v1dv2 - v0dv1 * v0dv2)/det
      );

      return bary;
   }

   // Returns the point on triangle [a, b, c] closest to point q.
   CUDA_CALLABLE pointBaryCoord_t closestPointToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & q
   )
   {
      pointBaryCoord_t bary_point;

      const cumath::Vector3 n((b - a).crossProduct(c - a));

      bary_point.point = cugeom::plane::closestPointToPoint(n, c, q);
      bary_point.baryCoords = baryCoords(a, b, c, bary_point.point);

      return bary_point;
   }

   // Returns the Voronoi region on the triangle [a, b, c] containing the
   // point q. This is also the minimum simplex closest to point q.
   // This could definitely be parallelized.
   CUDA_CALLABLE simplex3_t voronoiRegionToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & q
   )
   {
      cumath::Vector3 triangle[3] = {a, b, c};

      simplex3_t vertex_region;
      vertex_region.numVerts = 0;
      simplex3_t edge_region;
      edge_region.numVerts = 0;
      simplex3_t face_region;

      face_region.numVerts = 3;
      face_region.vertIdxs[0] = 0;
      face_region.vertIdxs[1] = 1;
      face_region.vertIdxs[2] = 2;
      face_region.verts[0] = triangle[0];
      face_region.verts[1] = triangle[1];
      face_region.verts[2] = triangle[2];

      // Check vertices
      for (int i = 0; i < 3; ++i)
      {
         cumath::Vector3 & vert_a = triangle[i];
         cumath::Vector3 & vert_b = triangle[(i + 1) % 3];
         cumath::Vector3 & vert_c = triangle[(i + 2) % 3];
         if (
            ((q - vert_a).dot(vert_b - vert_a) <= 0.0f) &&
            ((q - vert_a).dot(vert_c - vert_a) <= 0.0f)
         )
         {
            vertex_region.numVerts = 1;
            vertex_region.vertIdxs[0] = i;
            vertex_region.verts[0] = triangle[i];
         }
      }

      // Check edges
      for (int i = 0; i < 3; ++i)
      {
         cumath::Vector3 & vert_a = triangle[triangle_edge_combos[i].vertAId];
         cumath::Vector3 & vert_b = triangle[triangle_edge_combos[i].vertBId];
         cumath::Vector3 & vert_c = triangle[triangle_edge_combos[i].vertCId];

         cumath::Vector3 nABC = (vert_b - vert_a).crossProduct(vert_c - vert_a);
         if (
            ((q - vert_a).dot(vert_b - vert_a) >= 0.0f) &&
            ((q - vert_b).dot(vert_a - vert_b) >= 0.0f) &&
            ((q - vert_a).dot((vert_b - vert_a).crossProduct(nABC)) >= 0.0f)
         )
         {
            edge_region.numVerts = 2;
            edge_region.vertIdxs[0] = triangle_edge_combos[i].vertAId;
            edge_region.vertIdxs[1] = triangle_edge_combos[i].vertBId;
            edge_region.verts[0] = vert_a;
            edge_region.verts[1] = vert_b;
         }
      }

      return (
         (vertex_region.numVerts == 1) ? vertex_region : (
            (edge_region.numVerts == 2) ? edge_region : (
               face_region
            )
         )
      );
   }

   CUDA_CALLABLE cumath::Vector3 calcNormal(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & center
   )
   {
      cumath::Vector3 normal = (b - a).crossProduct(c - a);
      normal *= (((a - center).dot(normal) >= 0.f) ? 1.f : -1.f);
      return normal;
   }

}

}
