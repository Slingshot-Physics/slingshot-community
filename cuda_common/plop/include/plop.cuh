#ifndef PLOP_CUDA_HEADER
#define PLOP_CUDA_HEADER

#include "defines.cuh"

#include "device_array.cuh"
#include "host_array.cuh"

namespace plop
{
   template <typename ArrayType>
   HOST_CALLABLE bool copy(
      const ArrayType * device_start,
      const ArrayType * device_end,
      ArrayType * host_start
   )
   {
      cudaError_t copy_error = cudaMemcpy(
         host_start,
         device_start,
         (device_end - device_start) * sizeof(ArrayType),
         cudaMemcpyDeviceToHost
      );

      return (copy_error == cudaSuccess);
   }
}

#endif
