#ifndef DEVICE_ARRAY_CUDA_HEADER
#define DEVICE_ARRAY_CUDA_HEADER

#include "defines.cuh"

#include "host_array.cuh"

#include <stdio.h>

namespace plop
{

   template <typename ArrayType, unsigned int SIZE>
   class DeviceArray
   {
      public:
         typedef ArrayType * iterator;

         typedef const ArrayType * const_iterator;

         HOST_CALLABLE DeviceArray(void)
            : size_(SIZE)
            , alloc_success_(true)
         {
            cudaError_t alloc_error = cudaMalloc((void **) &base_, sizeof(ArrayType) * size_);

            if (alloc_error != cudaSuccess)
            {
               printf(
                  "Couldn't allocate memory of size %d\n",
                  (size_ * sizeof(ArrayType))
               );
               alloc_success_ = false;
            }
         }

         CUDA_CALLABLE DeviceArray(ArrayType * base)
            : size_(SIZE)
            , base_(base)
            , alloc_success_(true)
         { }

         CUDA_CALLABLE DeviceArray(const DeviceArray<ArrayType, SIZE> & other)
            : size_(other.size_)
            , base_(other.base_)
            , alloc_success_(other.alloc_success_)
         { }

         CUDA_HOST_CALLABLE ~DeviceArray(void)
         { }

         CUDA_HOST_CALLABLE unsigned int size(void) const
         {
            return size_;
         }

         CUDA_HOST_CALLABLE unsigned int size_bytes(void) const
         {
            return size_ * sizeof(ArrayType);
         }

         CUDA_HOST_CALLABLE ArrayType * begin(void)
         {
            return base_;
         }

         CUDA_HOST_CALLABLE ArrayType * end(void)
         {
            return base_ + size_;
         }

         CUDA_HOST_CALLABLE const ArrayType * begin(void) const
         {
            return base_;
         }

         CUDA_HOST_CALLABLE const ArrayType * end(void) const
         {
            return base_ + size_;
         }

         CUDA_HOST_CALLABLE void clear(void)
         {
            cudaFree(base_);
         }

         HOST_CALLABLE DeviceArray<ArrayType, SIZE> & operator=(
            const HostArray<ArrayType, SIZE> & other
         )
         {
            cudaError_t copy_error = cudaMemcpy(
               begin(),
               other.begin(),
               other.size_bytes(),
               cudaMemcpyHostToDevice
            );

            if (copy_error != cudaSuccess)
            {
               printf("Error copying data from host array\n");
            }

            return *this;
         }

         CUDA_CALLABLE ArrayType & operator[](unsigned int i)
         {
            return base_[i];
         }

         CUDA_CALLABLE const ArrayType & operator[](unsigned int i) const
         {
            return base_[i];
         }

      private:

         ArrayType * base_;

         unsigned int size_;

         bool alloc_success_;

   };

}

#endif
