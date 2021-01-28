#ifndef TRIANGLE_CUDA_HEADER
#define TRIANGLE_CUDA_HEADER

#include "defines.cuh"

#include "geometry_types.cuh"

namespace cugeom
{

namespace triangle
{

   typedef cugeom::types::pointBaryCoord_t pointBaryCoord_t;
   typedef cugeom::types::simplex3_t simplex3_t;

   // For calculating Voronoi regions.
   struct edge_t
   {
      int vertAId;
      int vertBId;
      int vertCId;
   };

   CUDA_CALLABLE cumath::Vector3 baryCoords(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & q
   );

   // Returns the point on triangle [a, b, c] closest to point q. This function
   // assumes that one of the triangle's faces is the Voronoi region containing
   // the query point.
   CUDA_CALLABLE pointBaryCoord_t closestPointToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & q
   );

   // Returns the Voronoi region on the triangle [a, b, c] containing the
   // point. The Voronoi region is a sub-simplex of the triangle.
   CUDA_CALLABLE simplex3_t voronoiRegionToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & q
   );

   // Calculates the normal of a triangle [a, b, c] given the triangle vertices
   CUDA_CALLABLE cumath::Vector3 calcNormal(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & center
   );

}

}

#endif
