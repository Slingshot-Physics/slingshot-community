#ifndef HEAP_MATRIX_HEADER
#define HEAP_MATRIX_HEADER

#include <cassert>
#include <iomanip>

// Heap-allocated matrix type. The main buffer is a flat array.
class HeapMatrix
{

public:
   HeapMatrix(unsigned int numRows, unsigned int numCols);

   HeapMatrix(const HeapMatrix & mat);

   ~HeapMatrix(void);

   void eye(void);

   // Sets all elements of the buffer to zero.
   void Initialize(void);

   bool hasNan(void) const;

   unsigned int paddedSize(void) const;

   float operator()(unsigned int i, unsigned int j) const;

   float & operator()(unsigned int i, unsigned int j);

   // Calculates the inner product between a column vector and a row of this
   // matrix. This is equivalent to:
   //    A.row(i).dot(vec)
   // Calculates:
   //
   //    for i = 1 to num_cols:
   //       val += A(row_a, i) * vec(0, i)
   //
   float innerProductRowVec(unsigned int row_a, const HeapMatrix & vec) const;

   // Does the same as the innerProductRowVec, except that SSE instructions are
   // used to speed up the calculation.
   float innerProductRowVecSse(
      unsigned int row_a, const HeapMatrix & vec
   ) const;

   // Calculates the inner product between a column of this matrix and a column
   // vector. This is equivalent to:
   //    (A ^ T).row(j).dot(vec)
   //    or
   //    A.col(j).dot(vec)
   float innerProductColVec(unsigned int col, HeapMatrix & vec) const;

   // Calculates the inner product between a sub-row of this matrix and a
   // sub-row of another matrix. The sub-row of this matrix starts at
   // (row_a, col_a) and the sub-row of 'mat' starts at (row_b, col_b).
   // 'size' is the length of the vectors being contracted.
   // Calculates:
   //
   //    for i = 1 to size:
   //       val += A(row_a + i, col_a) * B(row_b + i, col_b)
   //
   float innerProductSubRowMatRow(
      unsigned int row_a,
      unsigned int col_a,
      unsigned int row_b,
      unsigned int col_b,
      const HeapMatrix & mat,
      unsigned int size
   ) const;

   // Calculates the inner product between a sub-row of this matrix and a
   // sub-row of another matrix. The sub-row of this matrix starts at
   // (row_a, col_a) and the sub-row of 'mat' starts at (row_b, col_b).
   // 'size' is the length of the vectors being contracted.
   // Calculates:
   //
   //    for i = 1 to size:
   //       val += A(row_a + i, col_a) * B(row_b + i, col_b)
   //
   float innerProductSubRowMatRowSse(
      unsigned int row_a,
      unsigned int col_a,
      unsigned int row_b,
      unsigned int col_b,
      const HeapMatrix & mat,
      unsigned int size
   ) const;

   // Returns a column matrix
   HeapMatrix row(unsigned int i) const;

   HeapMatrix transpose(void) const;

   HeapMatrix operator+(const HeapMatrix & other) const;

   HeapMatrix operator-(const HeapMatrix & other) const;

   HeapMatrix operator*(const HeapMatrix & other) const;

   HeapMatrix & operator=(const HeapMatrix & other);

   HeapMatrix & operator+=(const HeapMatrix & other);

   HeapMatrix & operator-=(const HeapMatrix & other);

   HeapMatrix operator+(float v) const;

   HeapMatrix operator-(float v) const;

   HeapMatrix operator*(float v) const;

   HeapMatrix operator/(float v) const;

   HeapMatrix & operator=(float v);

   HeapMatrix & operator+=(float v);

   HeapMatrix & operator-=(float v);

   HeapMatrix & operator*=(float v);

   HeapMatrix & operator/=(float v);

   bool operator==(const HeapMatrix & other) const;

   // More useful to have the dimensions publicly accessible but not publicly
   // modifiable.
   const unsigned int num_rows;

   const unsigned int num_cols;

private:

   const unsigned int num_padded_rows_;

   const unsigned int num_padded_cols_;

   const unsigned int mat_size_;

   alignas(16) float * base_;

   HeapMatrix(void);

   HeapMatrix(HeapMatrix & mat);

   unsigned int rowPadding(unsigned int numRows, unsigned int numCols)
   {
      unsigned int num_padded_rows = numRows;
      if (numRows == 1 && numCols != 1)
      {
         num_padded_rows = 1;
      }
      else
      {
         num_padded_rows = numRows + (numRows % 4 != 0) * (4 - numRows % 4);
      }

      return num_padded_rows;
   }

   unsigned int colPadding(unsigned int numRows, unsigned int numCols)
   {
      unsigned int num_padded_cols = numCols;
      if (numRows != 1 && numCols == 1)
      {
         num_padded_cols = 1;
      }
      else
      {
         num_padded_cols = numCols + (numCols % 4 != 0) * (4 - numCols % 4);
      }

      return num_padded_cols;
   }

   // Ambiguous operation - denied!
   HeapMatrix & operator*=(const HeapMatrix & other);

   // Converts a 2D matrix position into a single index for accessing data from
   // the buffer.
   unsigned int index(unsigned int i, unsigned int j) const
   {
      assert(i < num_padded_rows_);
      assert(j < num_padded_cols_);

      unsigned int idx = (i * num_padded_cols_ + j);
      assert(idx < mat_size_);

      return idx;
   }
};

std::ostream & operator<<(std::ostream & out, const HeapMatrix & A);

HeapMatrix operator*(float v, const HeapMatrix & A);

HeapMatrix operator/(float v, const HeapMatrix & A);

HeapMatrix operator+(float v, const HeapMatrix & A);

HeapMatrix operator-(float v, const HeapMatrix & A);

// Solves a MLCP using Projected Gauss-Seidel.
// A * x = b, x_min <= x <= x_max
// 'A' is a square matrix, N x N.
// 'b' is a vector, N x 1.
// 'x_bounds' is a rectangular matrix, N x 2.
//    lower_bound of x(n) = x_bounds(n, 0)
//    upper_bound of x(n) = x_bounds(n, 1)
// 'x' is a vector, N x 1.
// 'max_iters' is the maximum number of iterations to perform.
// 'threshold' is max size of the L2-norm of successive solutions that allows
// early termination.
void ProjectedGaussSeidel(
   const HeapMatrix & A,
   const HeapMatrix & b,
   const HeapMatrix & x_bounds,
   HeapMatrix & x,
   unsigned int max_iters = 100,
   float threshold = 1e-7f
);

#endif
