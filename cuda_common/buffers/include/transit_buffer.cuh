#ifndef TRANSIT_BUFFER_CUDA_HEADER
#define TRANSIT_BUFFER_CUDA_HEADER

#include "defines.cuh"

#include <stdio.h>

#include "buffer_wrapper.cuh"

// The TransitBuffer keeps track of a buffer on the host and on the device.
// It's designed to allocate a fixed-size host-side array, access elements in
// the host-side array, allocate a corresponding device-side buffer in global
// memory, and copy host --> device or device --> host.
// The TransitBuffer frees the device buffer at destruction.
template <typename BufferType, unsigned int SIZE>
class TransitBuffer
{
   public:
      HOST_CALLABLE TransitBuffer(void)
         : size_(SIZE)
      {
         cudaError_t alloc_err = cudaMalloc((void **) &device_buff_, sizeof(BufferType) * size_);

         if (alloc_err != cudaSuccess)
         {
            printf(
               "Couldn't allocate memory of size %d\n",
               (size_ * sizeof(BufferType))
            );
            alloc_success_ = false;
         }

         device_buffer_wrapper_.initialize(device_buff_, size_);

         host_buffer_wrapper_.initialize(host_buff_, size_);

         alloc_success_ = true;
      }

      HOST_CALLABLE ~TransitBuffer(void)
      {
         if (!alloc_success_)
         {
            return;
         }

         cudaFree(device_buff_);
         device_buff_ = nullptr;
      }

      HOST_CALLABLE BufferType & operator[](unsigned int i)
      {
         return host_buff_[i];
      }

      HOST_CALLABLE const BufferType & operator[](unsigned int i) const
      {
         return host_buff_[i];
      }

      HOST_CALLABLE bool copy_host_to_device(void)
      {
         if (!alloc_success_)
         {
            return false;
         }

         cudaError_t cpy_err = cudaMemcpy(
            device_buff_,
            host_buff_,
            size_ * sizeof(BufferType),
            cudaMemcpyHostToDevice
         );

         if (cpy_err != cudaSuccess)
         {
            return false;
         }

         return true;
      }

      HOST_CALLABLE bool copy_device_to_host(void)
      {
         if (!alloc_success_)
         {
            return false;
         }

         cudaError_t cpy_err = cudaMemcpy(
            host_buff_,
            device_buff_,
            size_ * sizeof(BufferType),
            cudaMemcpyDeviceToHost
         );

         if (cpy_err != cudaSuccess)
         {
            return false;
         }

         return true;
      }

      HOST_CALLABLE BufferWrapper<BufferType> device_buffer_wrapper(void)
      {
         return device_buffer_wrapper_;
      }

      HOST_CALLABLE BufferWrapper<BufferType> host_buffer_wrapper(void)
      {
         return host_buffer_wrapper_;
      }

      HOST_CALLABLE unsigned int size_bytes(void)
      {
         return size_ * sizeof(BufferType);
      }

      HOST_CALLABLE unsigned int size(void)
      {
         return size_;
      }

   private:
      unsigned int size_;

      bool alloc_success_;

      BufferType host_buff_[SIZE];

      BufferType * device_buff_;

      BufferWrapper<BufferType> host_buffer_wrapper_;

      BufferWrapper<BufferType> device_buffer_wrapper_;

};

#endif
