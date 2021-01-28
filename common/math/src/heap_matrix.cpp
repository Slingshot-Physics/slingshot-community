#include "heap_matrix.hpp"

#ifdef BUILD_SSE
#include "simd_ops.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

HeapMatrix::HeapMatrix(unsigned int numRows, unsigned int numCols)
   : num_rows(numRows)
   , num_cols(numCols)
   , num_padded_rows_(rowPadding(numRows, numCols))
   , num_padded_cols_(colPadding(numRows, numCols))
   , mat_size_(num_padded_cols_ * num_padded_rows_)
{
   base_ = new float[mat_size_];
   Initialize();
}

HeapMatrix::HeapMatrix(const HeapMatrix & mat)
   : num_rows(mat.num_rows)
   , num_cols(mat.num_cols)
   , num_padded_rows_(num_rows + (num_rows % 4 != 0) * (4 - num_rows % 4))
   , num_padded_cols_(num_cols + (num_cols > 1) * (num_cols % 4 != 0) * (4 - num_cols % 4))
   , mat_size_(num_padded_cols_ * num_padded_rows_)
{
   base_ = new float[mat_size_];
   Initialize();
   std::memcpy(base_, mat.base_, sizeof(float) * mat_size_);
}

HeapMatrix::~HeapMatrix(void)
{
   delete [] base_;
}

void HeapMatrix::Initialize(void)
{
   std::fill(base_, base_ + mat_size_, 0.0f);
}

bool HeapMatrix::hasNan(void) const
{
   for (unsigned int i = 0; i < mat_size_; ++i)
   {
      if (std::isnan(base_[i]))
      {
         return true;
      }
   }

   return false;
}

unsigned int HeapMatrix::paddedSize(void) const
{
   return num_padded_cols_ * num_padded_rows_;
}

void HeapMatrix::eye(void)
{
   Initialize();
   for (unsigned int i = 0; i < std::min(num_rows, num_cols); ++i)
   {
      (*this)(i, i) = 1.0f;
   }
}

float HeapMatrix::operator()(unsigned int i, unsigned int j) const
{
   assert(i < num_rows);
   assert(j < num_cols);

   return base_[index(i, j)];
}

float & HeapMatrix::operator()(unsigned int i, unsigned int j)
{
   assert(i < num_rows);
   assert(j < num_cols);
   return base_[index(i, j)];
}

float HeapMatrix::innerProductRowVec(
   unsigned int i, const HeapMatrix & vec
) const
{
   assert(i < num_rows);
   assert(vec.num_cols == 1);
   assert(vec.num_rows == num_cols);

   float val = 0.f;

   unsigned int idx_offset = i * num_padded_cols_;

   for (unsigned int j = 0; j < num_cols; ++j)
   {
      val += base_[idx_offset + j] * vec.base_[j];
   }

   return val;
}

float HeapMatrix::innerProductRowVecSse(
   unsigned int i, const HeapMatrix & vec
) const
{
   assert(i < num_rows);
   assert(vec.num_cols == 1);
   assert(vec.num_rows == num_cols);

#ifdef BUILD_SSE
   float val = 0.f;
   if (num_padded_cols_ < 4)
   {
      val = innerProductRowVec(i, vec);
   }
   else
   {
      dot_product(&(base_[index(i, 0)]), &(vec.base_[0]), num_padded_cols_, &val);
   }
#else
   float val = innerProductRowVec(i, vec);
#endif
   return val;
}

float HeapMatrix::innerProductSubRowMatRow(
   unsigned int row_a,
   unsigned int col_a,
   unsigned int row_b,
   unsigned int col_b,
   const HeapMatrix & mat,
   unsigned int size
) const
{
   assert(row_a < num_rows);
   assert(row_b < mat.num_rows);
   assert(mat.num_cols == num_cols);

   float val = 0.f;
   for (unsigned int j = 0; j < size; ++j)
   {
      val += base_[index(row_a, col_a + j)] * mat(row_b, col_b + j);
   }

   return val;
}

float HeapMatrix::innerProductSubRowMatRowSse(
   unsigned int row_a,
   unsigned int col_a,
   unsigned int row_b,
   unsigned int col_b,
   const HeapMatrix & mat,
   unsigned int size
) const
{
   assert(row_a < num_rows);
   assert(row_b < mat.num_rows);
   assert(mat.num_cols == num_cols);

#ifdef BUILD_SSE
   float val = 0.f;
   masked_dot_product(
      &(base_[index(row_a, col_a)]), &(mat.base_[mat.index(row_b, col_b)]), size, &val
   );
#else
   float val = innerProductSubRowMatRow(row_a, col_a, row_b, col_b, mat, size);
#endif

   return val;
}

float HeapMatrix::innerProductColVec(unsigned int j, HeapMatrix & vec) const
{
   assert(j < num_cols);
   assert(vec.num_cols == 1);
   assert(vec.num_rows == num_rows);

   float val = 0.f;

   for (unsigned int i = 0; i < num_rows; ++i)
   {
      val += base_[index(i, j)] * vec.base_[i];
   }

   return val;
}

HeapMatrix HeapMatrix::row(unsigned int i) const
{
   assert(i < num_rows);
   HeapMatrix R(1, num_cols);

   for (unsigned int j = 0; j < num_cols; ++j)
   {
      R(0, j) = (*this)(i, j);
   }

   return R;
}

HeapMatrix HeapMatrix::transpose(void) const
{
   HeapMatrix T(num_cols, num_rows);
   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         T(j, i) = (*this)(i, j);
      }
   }

   return T;
}

HeapMatrix HeapMatrix::operator+(const HeapMatrix & other) const
{
   assert(num_rows == other.num_rows);
   assert(num_cols == other.num_cols);

   HeapMatrix S(num_rows, num_cols);
   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         S(i, j) = (*this)(i, j) + other(i, j);
      }
   }

   return S;
}

HeapMatrix HeapMatrix::operator-(const HeapMatrix & other) const
{
   assert(num_cols == other.num_cols);
   assert(num_rows == other.num_rows);

   HeapMatrix S(num_rows, num_cols);
   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         S(i, j) = (*this)(i, j) - other(i, j);
      }
   }

   return S;
}

HeapMatrix HeapMatrix::operator*(const HeapMatrix & other) const
{
   assert(num_cols == other.num_rows);

   HeapMatrix S(num_rows, other.num_cols);
   float tempRowVal = 0.0f;

   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < other.num_cols; ++j)
      {
         for (unsigned int k = 0; k < num_cols; ++k)
         {
            tempRowVal += (*this)(i, k) * other(k, j);
         }
         S(i, j) = tempRowVal;
         tempRowVal = 0.0f;
      }
   }

   return S;
}

HeapMatrix & HeapMatrix::operator=(const HeapMatrix & other)
{
   assert(num_rows == other.num_rows);
   assert(num_cols == other.num_cols);

   std::memcpy(base_, other.base_, sizeof(float) * mat_size_);

   return *this;
}

HeapMatrix & HeapMatrix::operator+=(const HeapMatrix & other)
{
   assert(num_rows == other.num_rows);
   assert(num_cols == other.num_cols);

   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         (*this)(i, j) += other(i, j);
      }
   }

   return *this;
}

HeapMatrix & HeapMatrix::operator-=(const HeapMatrix & other)
{
   assert(num_rows == other.num_rows);
   assert(num_cols == other.num_cols);

   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         (*this)(i, j) -= other(i, j);
      }
   }

   return *this;
}

HeapMatrix HeapMatrix::operator+(float v) const
{
   HeapMatrix S(*this);
   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         S(i, j) += v;
      }
   }

   return S;
}

HeapMatrix HeapMatrix::operator-(float v) const
{
   return operator+(-1.0f * v);
}

HeapMatrix HeapMatrix::operator*(float v) const
{
   HeapMatrix S(*this);
   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         S(i, j) *= v;
      }
   }

   return S;
}

HeapMatrix HeapMatrix::operator/(float v) const
{
   return operator*(1.0f / v);
}

HeapMatrix & HeapMatrix::operator=(float v)
{
   std::fill(base_, base_ + num_rows * num_cols, v);

   return *this;
}

HeapMatrix & HeapMatrix::operator+=(float v)
{
   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         (*this)(i, j) += v;
      }
   }

   return *this;
}

HeapMatrix & HeapMatrix::operator-=(float v)
{
   return operator+=(-1.0f * v);
}

HeapMatrix & HeapMatrix::operator*=(float v)
{
   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         (*this)(i, j) *= v;
      }
   }

   return *this;
}

HeapMatrix & HeapMatrix::operator/=(float v)
{
   return operator*=(1.0f / v);
}

bool HeapMatrix::operator==(const HeapMatrix & other) const
{
   if (other.num_rows != num_rows)
   {
      return false;
   }
   if (other.num_cols != num_cols)
   {
      return false;
   }

   for (unsigned int i = 0; i < num_rows; ++i)
   {
      for (unsigned int j = 0; j < num_cols; ++j)
      {
         if ((*this)(i, j) != other(i, j))
         {
            return false;
         }
      }
   }

   return true;
}

std::ostream & operator<< (std::ostream & out, const HeapMatrix & A)
{
   if (A.num_cols > 1)
   {
      for (unsigned int i = 0; i < A.num_rows; ++i)
      {
         for (unsigned int j = 0; j < A.num_cols; ++j)
         {
            out << std::setprecision(10) << std::setw(15) << std::right << A(i, j);
            if (j < A.num_cols - 1)
            {
               out << " ";
            }
            else
            {
               out << "\n";
            }
         }
      }
   }
   else
   {
      out << "[ ";
      for (unsigned int i = 0; i < A.num_rows; ++i)
      {
         out << std::setprecision(10) << std::setw(5) << std::right << A(i, 0);
         if (i < A.num_rows - 1)
         {
            out << " ";
         }
         else
         {
            out << "]\n";
         }
      }
   }
   return out;
}

HeapMatrix operator*(float v, const HeapMatrix & A)
{
   return A * v;
}

HeapMatrix operator/(float v, const HeapMatrix & A)
{
   return A * (1.0f / v);
}

HeapMatrix operator+(float v, const HeapMatrix & A)
{
   return A + v;
}

HeapMatrix operator-(float v, const HeapMatrix & A)
{
   return A + (-1.0f * v);
}

void ProjectedGaussSeidel(
   const HeapMatrix & A,
   const HeapMatrix & b,
   const HeapMatrix & x_bounds,
   HeapMatrix & x,
   unsigned int max_iters,
   float threshold
)
{
   HeapMatrix x_prev(x.num_rows, x.num_cols);

   x_prev += __FLT_MAX__;

   unsigned int iters = 0;
   for (iters = 0; iters < max_iters; ++iters)
   {
      for (unsigned int i = 0; i < x.num_rows; ++i)
      {
         const float & x_min = x_bounds(i, 0);
         const float & x_max = x_bounds(i, 1);
         float & x_val = x(i, 0);
         float delta_x = (b(i, 0) - A.innerProductRowVecSse(i, x)) / A(i, i);
         x_val = std::min(
            std::max(x_val + delta_x, x_min),
            x_max
         );
      }

      float distance = 0.0f;
      for (unsigned int j = 0; j < x.num_rows; ++j)
      {
         float element_diff = x_prev(j, 0) - x(j, 0);
         distance += element_diff * element_diff;
      }

      if (sqrtf(distance) < threshold)
      {
         break;
      }

      x_prev = x;
   }
}
