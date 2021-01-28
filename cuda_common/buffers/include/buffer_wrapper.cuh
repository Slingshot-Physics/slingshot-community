#ifndef BUFFER_WRAPPER_CUDA_HEADER
#define BUFFER_WRAPPER_CUDA_HEADER

#include "defines.cuh"

#include <stdio.h>

// The buffer wrapper can wrap around an existing buffer on the device or host
// or it can allocate a dynamic array on the host. A dynamically allocated
// array must be manually deleted by the user.
template <typename BufferType>
class BufferWrapper
{
   public:
      CUDA_HOST_CALLABLE BufferWrapper(
         BufferType * buff, unsigned int size
      )
         : size_(size)
         , buff_(buff)
      { }

      CUDA_HOST_CALLABLE BufferWrapper(void)
      {
         buff_ = nullptr;
         size_ = 0;
      }

      HOST_CALLABLE BufferWrapper(unsigned int size)
         : size_(size)
      {
         buff_ = new BufferType[size_];
      }

      CUDA_HOST_CALLABLE void initialize(BufferType * buff, unsigned int size)
      {
         size_ = size;
         buff_ = buff;
      }

      HOST_CALLABLE void clear(void)
      {
         delete[] buff_;
         buff_ = nullptr;
      }

      CUDA_HOST_CALLABLE BufferType & operator[](unsigned int i)
      {
         return buff_[i];
      }

      CUDA_HOST_CALLABLE unsigned int size(void)
      {
         return size_;
      }

   private:
      unsigned int size_;

      BufferType * buff_;
};

#endif
