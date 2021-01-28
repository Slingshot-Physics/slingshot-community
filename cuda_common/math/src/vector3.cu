#include "vector3.cuh"

namespace cumath
{

   CUDA_HOST_CALLABLE Vector3 operator* (const float & v, const Vector3 & a)
   {
      return a * v;
   }

}
