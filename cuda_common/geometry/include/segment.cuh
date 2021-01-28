#ifndef SEGMENT_CUDA_HEADER
#define SEGMENT_CUDA_HEADER

#include "defines.cuh"

#include "geometry_types.cuh"

namespace cugeom
{

namespace segment
{

   typedef cugeom::types::simplex3_t simplex3_t;
   typedef cugeom::types::pointBaryCoord_t pointBaryCoord_t;

   CUDA_CALLABLE float closestParameterToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & q
   );

   CUDA_CALLABLE pointBaryCoord_t closestPointToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & q
   );

   CUDA_CALLABLE simplex3_t voronoiRegionToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & q
   );

}

}

#endif
