#include "plane.cuh"

namespace cugeom
{

namespace plane
{

   CUDA_CALLABLE cumath::Vector3 closestPointToPoint(
      const cumath::Vector3 & normal,
      const cumath::Vector3 & a,
      const cumath::Vector3 & q
   )
   {
      float t = normal.dot(q - a)/normal.magnitudeSquared();
      cumath::Vector3 p = q - t * normal;
      return p;
   }

}

}
