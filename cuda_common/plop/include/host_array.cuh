#ifndef PLOP_HOST_ARRAY_CUDA_HEADER
#define PLOP_HOST_ARRAY_CUDA_HEADER

#include "defines.cuh"

namespace plop
{

   template <typename ArrayType, unsigned int SIZE>
   class HostArray
   {
      public:
         typedef const ArrayType * const_iterator;

         typedef ArrayType * iterator;

         HOST_CALLABLE HostArray(void)
            : size_(SIZE)
         { }

         HOST_CALLABLE unsigned int size(void) const
         {
            return size_;
         }

         HOST_CALLABLE unsigned int size_bytes(void) const
         {
            return size_ * sizeof(ArrayType);
         }

         HOST_CALLABLE ArrayType * begin(void)
         {
            return &(base_[0]);
         }

         HOST_CALLABLE ArrayType * end(void)
         {
            return &(base_[0]) + SIZE;
         }

         HOST_CALLABLE const ArrayType * begin(void) const
         {
            return &(base_[0]);
         }

         HOST_CALLABLE const ArrayType * end(void) const
         {
            return &(base_[0]) + SIZE;
         }

         HOST_CALLABLE ArrayType & operator[](unsigned int i)
         {
            return base_[i];
         }

         HOST_CALLABLE const ArrayType & operator[](unsigned int i) const
         {
            return base_[i];
         }

         HOST_CALLABLE ArrayType & operator=(
            const HostArray<ArrayType, SIZE> & other
         )
         {
            for (unsigned int i = 0; i < size_; ++i)
            {
               base_[i] = other.base_[i];
            }

            return *this;
         }

      private:

         ArrayType base_[SIZE];

         unsigned int size_;
   };

}

#endif
