#ifndef SIMPLEX_CUDA_HEADER
#define SIMPLEX_CUDA_HEADER

#include "defines.cuh"

#include "geometry_types.cuh"

namespace cugeom
{

namespace simplex
{

   typedef cugeom::types::simplex3_t simplex3_t;
   typedef cugeom::types::pointBaryCoord_t pointBaryCoord_t;

   CUDA_CALLABLE simplex3_t voronoiRegionToPoint(
      const simplex3_t & simplex, const cumath::Vector3 & q
   );

   CUDA_CALLABLE pointBaryCoord_t closestPointToPoint(
      const simplex3_t & simplex, const cumath::Vector3 & q
   );

}

}

#endif
