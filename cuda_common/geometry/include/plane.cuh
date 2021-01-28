#ifndef PLANE_CUDA_HEADER
#define PLANE_CUDA_HEADER

#include "defines.cuh"

#include "vector3.cuh"

namespace cugeom
{

namespace plane
{

   CUDA_CALLABLE cumath::Vector3 closestPointToPoint(
      const cumath::Vector3 & normal,
      const cumath::Vector3 & a,
      const cumath::Vector3 & q
   );

}

}

#endif
