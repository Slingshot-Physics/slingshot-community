#ifndef DYNAMIC_ARRAY_HEADER
#define DYNAMIC_ARRAY_HEADER

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

template <typename ArrayType>
class DynamicArray
{
   public:

      typedef const ArrayType * const_iterator;

      typedef ArrayType * iterator;

      DynamicArray(unsigned int capacity)
         : size_(0)
         , capacity_(capacity)
         , delta_capacity_(32)
      {
         base_ = new ArrayType[capacity];
      }

      DynamicArray(unsigned int capacity, ArrayType default_val)
         : size_(0)
         , capacity_(capacity)
         , delta_capacity_(32)
      {
         base_ = new ArrayType[capacity];
         for (unsigned int i = 0; i < capacity; ++i)
         {
            push_back(default_val);
         }
      }

      ~DynamicArray(void)
      {
         delete[] base_;
         base_ = nullptr;
      }

      ArrayType & operator[](std::size_t i)
      {
         return base_[i];
      }

      const ArrayType & operator[](std::size_t i) const
      {
         return base_[i];
      }

      ArrayType & at(std::size_t i)
      {
         if (i >= size_)
         {
            throw std::invalid_argument("Index out of bounds");
         }

         return base_[i];
      }

      const ArrayType & at(std::size_t i) const
      {
         if (i >= size_)
         {
            throw std::invalid_argument("Index out of bounds");
         }

         return base_[i];
      }

      ArrayType * base_pointer(void)
      {
         return &(base_[0]);
      }

      // Returns the first element of the array.
      ArrayType & front(void)
      {
         return base_[0];
      }

      // Returns the first element of the array.
      const ArrayType & front(void) const
      {
         return base_[0];
      }

      // Returns the last element of the array.
      ArrayType & back(void)
      {
         return base_[size_ - 1];
      }

      // Returns the last element of the array.
      const ArrayType & back(void) const
      {
         return base_[size_ - 1];
      }

      iterator begin(void)
      {
         return &(base_[0]);
      }

      iterator end(void)
      {
         iterator temp_end = &(base_[0]) + size_;
         return temp_end;
      }

      const_iterator begin(void) const
      {
         return &(base_[0]);
      }

      const_iterator end(void) const
      {
         iterator temp_end = &(base_[0]) + size_;
         return temp_end;
      }

      void append(const ArrayType & new_val)
      {
         if (size_ + 1 >= capacity_)
         {
            increase_capacity();
         }

         base_[size_] = new_val;
         size_ += 1;
      }

      void push_back(const ArrayType & new_val)
      {
         append(new_val);
      }

      ArrayType pop(std::size_t i)
      {
         ArrayType val = base_[i];

         if (i < size_ - 1)
         {
            std::move(&base_[i + 1], &base_[size_], &base_[i]);
         }
         size_ -= 1;

         return val;
      }

      // Deletes the last element in the array and returns the last element in
      // the array.
      ArrayType pop_back(void)
      {
         ArrayType val = base_[size_ - 1];

         if (size_ > 0)
         {
            size_ -= 1;
         }

         return val;
      }

      void clear(void)
      {
         size_ = 0;
      }

      std::size_t size(void) const
      {
         return size_;
      }

      std::size_t capacity(void) const
      {
         return capacity_;
      }

   private:

      DynamicArray(const DynamicArray &);

      DynamicArray(DynamicArray &);

      void increase_capacity(void)
      {
         ArrayType * temp_arr = new ArrayType[capacity_ + delta_capacity_];
         for (unsigned int i = 0; i < size_; ++i)
         {
            temp_arr[i] = base_[i];
         }
         delete[] base_;
         base_ = temp_arr;
         capacity_ += delta_capacity_;
      }

      ArrayType * base_;

      std::size_t size_;

      std::size_t capacity_;

      const std::size_t delta_capacity_;
};

#endif
