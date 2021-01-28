#ifndef TETRAHEDRON_CUDA_HEADER
#define TETRAHEDRON_CUDA_HEADER

#include "defines.cuh"

#include "geometry_types.cuh"

namespace cugeom
{

namespace tetrahedron
{

   typedef cugeom::types::simplex3_t simplex3_t;
   typedef cugeom::types::pointBaryCoord_t pointBaryCoord_t;

   // Vertices A and B are the edge vertices. C and D are the remaining
   // vertices.
   struct edge_t
   {
      int vertAId;
      int vertBId;
      int vertCId;
      int vertDId;
   };

   // First three vertices form the face.
   struct face_t
   {
      int vertAId;
      int vertBId;
      int vertCId;
      int vertDId;
   };

   // Finds the closest point in the tetrahedron to the query point. This
   // function assumes that the tetrahedron [a, b, c, d] is the minimum simplex
   // containing the query point, q.
   CUDA_CALLABLE pointBaryCoord_t closestPointToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & d,
      const cumath::Vector3 & q
   );

   // Calculates potential Voronoi regions for all combinations of
   // sub-simplices of the tetrahedron defined by [a, b, c, d]. The correct
   // simplex is selected and returned.  The Voronoi region is a sub-simplex
   // of the tetrahedron.
   CUDA_CALLABLE simplex3_t voronoiRegionToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & d,
      const cumath::Vector3 & q
   );

}

}

#endif
