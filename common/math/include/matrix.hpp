#ifndef MATH_MATRIX_HEADER
#define MATH_MATRIX_HEADER

#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <iomanip>

#include "matrix33.hpp"
#include "vector3.hpp"

template <unsigned int M, unsigned int N>
class Matrix
{
   public:
      Matrix (float v)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            std::memset(base_[i], v, sizeof(base_[i]));
         }
      }

      Matrix (unsigned int n, ...)
      {
         va_list va;
         va_start(va, n);
         unsigned int row = 0;
         unsigned int column = 0;
         for (unsigned int i = 0; i < std::min(n, M * N); ++i)
         {
            row = i / N;
            column = i % N;
            base_[row][column] = (float )va_arg(va, double);
         }
         va_end(va);
      }

      Matrix (const float arr[M])
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] = arr[i];
            }
         }
      }

      Matrix (const Matrix<M, N> & A)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            // std::memcpy(base_[i], A.base_[i], sizeof(base_[i]));
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] = A(i, j);
            }
         }
      }

      Matrix (void)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] = 0.0f;
               // std::memset(base_[i], 0, sizeof(base_[i]));
            }
         }
      }

      Matrix<N, M> transpose (void) const
      {
         Matrix<N, M> T;
         for (unsigned int i = 0; i < N; ++i)
         {
            for (unsigned int j = 0; j < M; ++j)
            {
               T(i, j) = base_[j][i];
            }
         }

         return T;
      }

      bool hasNan(void) const
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               if (std::isnan(base_[i][j]))
               {
                  return true;
               }
            }
         }

         return false;
      }

      void assignCol(unsigned int col, const float colArr[M])
      {
         assert(col < N);
         for (unsigned int i = 0; i < M; ++i)
         {
            base_[i][col] = colArr[i];
         }
      }

      void assignCol(unsigned int col, const Matrix<M, 1> & colArr)
      {
         assert(col < N);
         for (unsigned int i = 0; i < M; ++i)
         {
            base_[i][col] = colArr(i, 0);
         }
      }

      void assignRow(unsigned int row, const float rowArr[N])
      {
         assert (row < M); 
         for (unsigned int j = 0; j < N; ++j)
         {
            base_[row][j] = rowArr[j];
         }
      }

      void assignRow(unsigned int row, const Matrix<N, 1> & rowArr)
      {
         assert (row < M);
         for (unsigned int j = 0; j < N; ++j)
         {
            base_[row][j] = rowArr(j, 0);
         }
      }

      // Slice is based on the row and column designation (top-left corner of
      // the slice) and the size of the matrix being assigned.
      template <unsigned int P, unsigned int Q>
      void assignSlice(
         unsigned int row, unsigned int col, const Matrix<P, Q> & S
      )
      {
         assert (row + P <= M);
         assert (col + Q <= N);

         for (unsigned int i = row; i < row + P; ++i)
         {
            for (unsigned int j = col; j < col + Q; ++j)
            {
               base_[i][j] = S(i - row, j - col);
            }
         }
      }

      void assignSlice(
         unsigned int row, unsigned int col, const Matrix33 & S
      )
      {
         unsigned int P = 3;
         unsigned int Q = 3;
         assert (row + P <= M);
         assert (col + Q <= N);
         for (unsigned int i = row; i < row + P; ++i)
         {
            for (unsigned int j = col; j < col + Q; ++j)
            {
               base_[i][j] = S(i - row, j - col);
            }
         }
      }

      void assignSlice(
         unsigned int row, unsigned int col, const Vector3 & v
      )
      {
         unsigned int P = 3;
         assert (row + P <= M);
         assert (col + 1 <= N);
         for (unsigned int i = row; i < row + P; ++i)
         {
            base_[i][col] = v[i - row];
         }
      }

      // You want it as a Matrix33 or Vector3? Go fuck yourself.
      template <unsigned int P, unsigned int Q>
      Matrix<P, Q> getSlice(unsigned int row, unsigned int col) const
      {
         assert (row + P <= M);
         assert (col + Q <= N);
         Matrix<P, Q> ret;
         for (unsigned int i = row; i < row + P; ++i)
         {
            for (unsigned int j = col; j < col + Q; ++j)
            {
               ret(i - row, j - col) = base_[i][j];
            }
         }

         return ret;
      }

      float determinant (void) const
      {
         if (M != N)
         {
            // You done fucked up A-a-ron
            return nan("");
         }

         float det = 0.0f;
         switch(M)
         {
            case 1:
            {
               det = base_[0][0];
               break;
            }
            case 2:
            {
               det = base_[0][0] * base_[1][1] - base_[0][1] * base_[1][0];
               break;
            }
            case 3:
            {
               det = \
                  base_[0][0]*base_[1][1]*base_[2][2] +
                  base_[0][1]*base_[1][2]*base_[2][0] +
                  base_[0][2]*base_[1][0]*base_[2][1] -
                  base_[2][0]*base_[1][1]*base_[0][2] -
                  base_[2][1]*base_[1][2]*base_[0][0] -
                  base_[2][2]*base_[1][0]*base_[0][1];
               break;
            }
            default:
            {
               Matrix<M, N> L;
               Matrix<M, N> U;
               LUDecomposition(*this, L, U);
               det = 1.0;
               for (unsigned int i = 0; i < M; ++i)
               {
                  det *= U(i, i);
               }

               break;
            }
         }
         return det;
      }

      void eye(void)
      {
         assert(M == N);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < M; ++j)
            {
               base_[i][j] = (i == j) * 1.0f + 0.0f;
            }
         }
      }

      float norm(void)
      {
         float val = 0.f;
         Matrix<M, M> symm = (*this) * (*this).transpose();
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < M; ++j)
            {
               val += symm(i, j);
            }
         }

         return sqrtf(val);
      }

      Matrix<N, 1> operator() (unsigned int i) const
      {
         Matrix<N, 1> ret(base_[i]);
         return ret;
      }

      float & operator() (unsigned int i, unsigned int j)
      {
         if (i >= M || j >= N)
         {
            // OOB? You get a crash.
            exit(-1);
         }
         return base_[i][j];
      }

      float operator() (unsigned int i, unsigned int j) const
      {
         if (i >= M || j >= N)
         {
            // OOB? You get a crash.
            exit(-1);
         }
         return base_[i][j];
      }

      template <unsigned int P, unsigned int Q>
      Matrix<M, N> & operator= (const Matrix<P, Q> & A)
      {
         if ((P != M) || (Q != 1) || (Q != N))
         {
            // Stop it, get some help.
            return *this;
         }

         float assignment = 0.0f;
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               if (Q == 1)
               {
                  assignment = A(i, 0);
               }
               else
               {
                  assignment = A(i, j);
               }
               base_[i][j] = assignment;
            }
         }

         return *this;
      }

      Matrix<M, N> & operator= (const Matrix<M, N> & A)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] = A.base_[i][j];
            }
         }

         return *this;
      }

      Matrix<M, N> & operator= (const float & v)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] = v;
            }
         }

         return *this;
      }

      Matrix<M, N> operator+ (const Matrix<M, N> & V) const
      {
         Matrix<M, N> result(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               result(i, j) += V(i, j);
            }
         }

         return result;
      }

      Matrix<M, N> operator+ (const float & v) const
      {
         Matrix<M, N> result(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               result(i, j) += v;
            }
         }

         return result;
      }

      Matrix<M, N> & operator+= (const float & v)
      {
         Matrix<M, N> result(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] += v;
            }
         }

         return *this;
      }

      Matrix<M, N> & operator+= (const Matrix<M, N> & V)
      {
         Matrix<M, N> result(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] += V(i, j);
            }
         }

         return *this;
      }

      Matrix<M, N> operator- (const float & v) const
      {
         Matrix<M, N> result(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               result(i, j) -= v;
            }
         }

         return result;
      }

      Matrix<M, N> operator- (const Matrix<M, N> & V) const
      {
         Matrix<M, N> result(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               result(i, j) -= V(i, j);
            }
         }

         return result;
      }

      Matrix<M, N> & operator-= (const float & v)
      {
         Matrix<M, N> result(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] -= v;
            }
         }

         return *this;
      }

      Matrix<M, N> & operator-= (const Matrix<M, N> & V)
      {
         Matrix<M, N> result(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] -= V(i, j);
            }
         }

         return *this;
      }

      Matrix<M, N> operator* (const float & v) const
      {
         Matrix<M, N> ret(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               ret(i, j) *= v;
            }
         }

         return ret;
      }

      template <unsigned int P>
      Matrix<M, P> operator* (const Matrix<N, P> & V) const
      {
         Matrix<M, P> ret;
         float tempRowVal = 0.0f;
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < P; ++j)
            {
               for (unsigned int k = 0; k < N; ++k)
               {
                  tempRowVal += base_[i][k] * V(k, j);
               }
               ret(i, j) = tempRowVal;
               tempRowVal = 0.0f;
            }
         }

         return ret;
      }

      Matrix<M, 1> operator* (const Matrix<N, 1> & V) const
      {
         Matrix<M, 1> ret;
         float tempRowVal = 0.0f;
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               tempRowVal += base_[i][j] * V(j, 0);
            }
            ret(i, 0) = tempRowVal;
            tempRowVal = 0.0f;
         }

         return ret;
      }

      Matrix<M, N> & operator*= (const float & v)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] *= v;
            }
         }

         return *this;
      }

      Matrix<M, N> & operator*= (const Matrix<1, 1> & V)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] *= V.base_[0][0];
            }
         }

         return *this;
      }

      Matrix<M, N> operator/ (const float & v) const
      {
         Matrix<M, N> ret(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               ret(i, j) /= v;
            }
         }

         return ret;
      }

      Matrix<M, N> operator/ (const Matrix<1, 1> & v) const
      {
         Matrix<M, N> ret(*this);
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               ret(i, j) /= v.base_[0][0];
            }
         }

         return ret;
      }

      Matrix<M, N> & operator/= (const float & v)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] /= v;
            }
         }

         return *this;
      }

      Matrix<M, N> & operator/= (const Matrix<1, 1> & v)
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               base_[i][j] /= v.base_[0][0];
            }
         }

         return *this;
      }

      bool operator== (const Matrix<M, N> & other) const
      {
         for (unsigned int i = 0; i < M; ++i)
         {
            for (unsigned int j = 0; j < N; ++j)
            {
               if (base_[i][j] != other.base_[i][j])
               {
                  return false;
               }
            }
         }

         return true;
      }

   private:
      // M rows, N columns.
      float base_[M][N];

      template <unsigned int P>
      Matrix<M, P> operator*= (const Matrix<N, P> & A);

};

template <unsigned int M>
Matrix<M, M> eye (void)
{
   Matrix<M, M> result;
   for (unsigned int i = 0; i < M; ++i)
   {
      result(i, i) = 1.0f;
   }

   return result;
}

template <unsigned int M>
void invGaussElim (
   const Matrix<M, M> & A, Matrix<M, M> & A_inv
)
{
   // Permutation matrices
   Matrix<M, M> perms[M];
   for (unsigned int i = 0; i < M; ++i)
   {
      perms[i] = eye<M>();
   }

   // E = [A | I]
   Matrix<M, 2 * M> E;
   for (unsigned int i = 0; i < M; ++i)
   {
      for (unsigned int j = 0; j < M; ++j)
      {
         E(i, j) = A(i, j);
         if (i == j)
         {
            E(i, j + M) = 1.0f;
         }
      }
   }

   const unsigned int N = 2 * M;
   Matrix<N, 1> r_i;
   Matrix<N, 1> r_n;
   float factor = 0.0f;
   // Go go gadget Gauss-Jordan Elimination (with diagonal normalization).
   for (unsigned int i = 0; i < M - 1; ++i)
   {
      // Permutations aren't being performed because I live dangerously.
      //    Edit: Why do I choose to be like this.
      E.assignRow(i, E(i) / E(i, i));
      r_i = E(i);
      for (unsigned int j = i + 1; j < M; ++j)
      {
         r_n = E(j);
         factor = r_n(i, 0);
         r_n = r_n - factor * r_i;
         E.assignRow(j, r_n);
      }
   }
   E.assignRow(M - 1, E(M - 1) / E(M - 1, M - 1));

   for (unsigned int i = M - 1; i >= 1; --i)
   {
      for (unsigned int j = 0; j < i; ++j)
      {
         E.assignRow(j, E(j) - E(j, i) * E(i));
      }
   }

   for (unsigned int i = 0; i < M; ++i)
   {
      for (unsigned int j = 0; j < M; ++j)
      {
         A_inv(i, j) = E(i, j + M);
      }
   }
}

template <unsigned int M>
Matrix<M, M> invGaussElim (const Matrix<M, M> & A)
{
   Matrix<M, M> ret;
   invGaussElim(A, ret);
   return ret;
}

template <unsigned int M>
void LUDecomposition (
   const Matrix<M, M> & A, Matrix<M, M> & L, Matrix<M, M> & U
)
{
   // Because I will forget which index refers to rows and which refers to
   // columns.
   const unsigned int N = M;
   L = eye<M>();
   U = A;

   Matrix<N, 1> r_i;
   Matrix<N, 1> r_n;
   float factor = 0.0f;
   float bottom_val = 0.0;
   // Go go gadget Gaussian Elimination.
   for (unsigned int i = 0; i < N - 1; ++i)
   {
      r_i = U(i);
      bottom_val = A(i, i);
      for (unsigned int j = i + 1; j < N; ++j)
      {
         r_n = U(j);
         factor = (r_n(i, 0)/bottom_val);
         r_n = r_n - factor * r_i;
         U.assignRow(j, r_n);
         L(j, i) = factor;
      }
   }
}

template <unsigned int M, unsigned int N>
Matrix<M, N> operator* (float v, const Matrix<M, N> & A)
{
   Matrix<M, N> ret(A);
   for (unsigned int i = 0; i < M; ++i)
   {
      for (unsigned int j = 0; j < N; ++j)
      {
         ret(i, j) *= v;
      }
   }

   return ret;
}

template <unsigned int M, unsigned int N>
Matrix<M, N> operator* (const Matrix<1, 1> & v, const Matrix<M, N> & A)
{
   Matrix<M, N> ret(A);
   for (unsigned int i = 0; i < M; ++i)
   {
      for (unsigned int j = 0; j < N; ++j)
      {
         ret(i, j) *= v(0, 0);
      }
   }

   return ret;
}

// Performs the multiplication A * C * A.T, where C and A are given.
// This function reduces the computational complexity of
// matrix multiplication by assuming C is block diagonal with some max
// block size. Note that the number of rows of C must be evenly divisible
// by the block size.
template <unsigned int B, unsigned int M, unsigned int N>
Matrix<M, M> ACAtBlockDiag(const Matrix<M, N> & A, const Matrix<N, N> & C)
{
   assert((N % B) == 0);
   unsigned int numBlocks = N / B;
   Matrix<M, M> ret(0.0f);
   Matrix<B, B> tempBlock;
   Matrix<M, B> tempSlice;

   for (unsigned int i = 0; i < numBlocks; ++i)
   {
      tempSlice = A.template getSlice<M, B>(0, i * B);
      tempBlock = C.template getSlice<B, B>(i * B, i * B);
      ret += (tempSlice * (tempBlock * tempSlice.transpose()));
   }

   return ret;
}

// Performs the multiplication A * C where A is square and is block-diagonal.
template <unsigned int B, unsigned int M, unsigned int P>
Matrix<M, P> blockMatmul(const Matrix<M, M> & A, const Matrix<M, P> & C)
{
   assert((M % B) == 0);
   unsigned int numBlocks = M / B;
   Matrix<M, P> ret(0.0f);
   Matrix<B, B> tempBlock;
   Matrix<B, P> tempSlice;
   for (unsigned int i = 0; i < numBlocks; ++i)
   {
      tempBlock = A.template getSlice<B, B>(i * B, i * B);
      tempSlice = C.template getSlice<B, P>(i * B, 0);
      ret.assignSlice(i * B, 0, tempBlock * tempSlice);
   }

   return ret;
}

// Generates a Givens/Jacobi rotation matrix for QR decomposition.
template <unsigned int M>
void givens(unsigned int row, float alpha, float beta, Matrix<M, M> & G)
{
   float gamma = 0.0f;
   float sigma = 0.0f;

   G.eye();
   if (fabs(alpha) > fabs(beta))
   {
      float tau = beta/alpha;
      gamma = 1.0 / sqrtf(1 + tau * tau);
      sigma = tau * gamma;
   }
   else
   {
      float tau = alpha/beta;
      sigma = 1.0 / sqrtf(1 + tau * tau);
      gamma = tau * sigma;
   }

   G(row - 1, row - 1) = gamma;
   G(row, row) = gamma;
   G(row - 1, row) = -1.0f * sigma;
   G(row, row - 1) = sigma;
}

// Generates a matrix inverse using QR factorization with Givens rotations.
template <unsigned int M>
void QRDecomposition(
   const Matrix<M, M> & A, Matrix<M, M> & Q, Matrix<M, M> & R
)
{
   // Q = eye<M>();
   Q.eye();
   R = A;
   Matrix<M, M> G;

   for (unsigned int j = 0; j < M; ++j)
   {
      for (unsigned int i = M - 1; i > j; --i)
      {
         if (fabs(R(i - 1, j)) > 1e-7 && fabs(R(i, j)) > 1e-7)
         {
            givens(i, R(i - 1, j), R(i, j), G);
            // Number of multiplies/accumulates can be reduced drastically because
            // most of the Givens matrix is identity.
            R = G.transpose() * R;
            // This should be identically zero, so this doesn't hurt.
            R(i, j) = 0.0f;
            Q = Q * G;
         }
      }
   }
}

// Generic back-substitution algorithm for an upper-triangular square matrix.
template <unsigned int M>
void backSubstitution(const Matrix<M, M> & R, Matrix<M, M> & S)
{
   Matrix<M, 2 * M> E;
   E.assignSlice(0, 0, R);
   E.assignSlice(0, M, eye<M>());
   // Matrix<2 * M, 1> moddyRow;
   float factor = 0.0f;

   // Set 1s in the diagonal.
   for (unsigned int i = 0; i < M; ++i)
   {
      factor = E(i, i);
      E.assignRow(i, E(i) / factor);
   }

   // Back-substitute using elementary row operations, starting with the bottom
   // right corner of the 1-diagonal matrix.
   for (unsigned int j = M - 1; j > 0; --j)
   {
      for (int i = j - 1; i > -1; --i)
      {
         factor = E(i, j);
         for (unsigned int k = 0; k < 2 * M; ++k){

            E(i, k) = E(i, k) - factor * E(j, k);
         }
         E(i, j) = 0.0f;
      }
   }

   for (unsigned int i = 0; i < M; ++i)
   {
      for (unsigned int j = 0; j < M; ++j)
      {
         S(i, j) = E(i, j + M);
      }
   }
}

// Performs the inverse of a QR-decomposed matrix, where Q is an orthogonal
// matrix and R is an upper triangular matrix.
template <unsigned int M>
void inverseQR(
   const Matrix<M, M> & Q, const Matrix<M, M> & R, Matrix<M, M> & E_inv
)
{
   Matrix<M, M> R_inv;
   backSubstitution(R, R_inv);
   E_inv = R_inv * Q.transpose();
}

// Performs an inverse of a matrix, A, using QR decomposition and back
// substitution. Convenience function.
template <unsigned int M>
Matrix<M, M> inverseQR(const Matrix<M, M> & A)
{
   Matrix<M, M> A_inv;
   Matrix<M, M> Q;
   Matrix<M, M> R;
   QRDecomposition(A, Q, R);
   inverseQR(Q, R, A_inv);
   return A_inv;
}

template <unsigned int M>
void inverseQR(
   const Matrix<M, M> & E, Matrix<M, M> & E_inv
)
{
   Matrix<M, M> Q, R;
   QRDecomposition(E, Q, R);
   Matrix<M, M> R_inv;
   backSubstitution(R, R_inv);
   E_inv = R_inv * Q.transpose();
}

template <unsigned int M>
void solveLinearSystem(const Matrix<M, M> & A, const Matrix<M, 1> & b, Matrix<M, 1> & x)
{
   Matrix<M, 1> x_prev(x);
   float tol = 1e-10;
   unsigned int maxIters = 100;
   unsigned int numIters = 0;

   // Lower and upper triangular matrices. L has the elements on the diagonal.
   Matrix<M, M> L, U, L_T_inv, L_inv;
   for (unsigned int i = 0; i < M; ++i)
   {
      for (unsigned int j = 0; j < M; ++j)
      {
         U(i, j) = A(i, j) * (i < j) + 0.0f;
         L(i, j) = A(i, j) * (i >= j) + 0.0f;
      }
   }

   // Calculate the inverse of L transpose
   backSubstitution(L.transpose(), L_T_inv);
   L_inv = L_T_inv.transpose();

   while (numIters < maxIters)
   {
      x = L_inv * (b - U * x_prev);
      if ((x - x_prev).norm() <= tol)
      {
         break;
      }
      x_prev = x;
      ++numIters;
   }
}

template <unsigned int M, unsigned int N>
std::ostream & operator<< (std::ostream & out, const Matrix<M, N> & A)
{
   if (N > 1)
   {
      for (unsigned int i = 0; i < M; ++i)
      {
         for (unsigned int j = 0; j < N; ++j)
         {
            out << std::setprecision(10) << std::setw(15) << std::right << A(i, j);
            if (j < N - 1)
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
      for (unsigned int i = 0; i < M; ++i)
      {
         out << std::setprecision(10) << std::setw(5) << std::right << A(i, 0);
         if (i < M - 1)
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

float determinant3x3(const Matrix<3, 3> & A);

Matrix<3, 3> inverse3x3(const Matrix<3, 3> & A);

Matrix<6, 6> inverse6x6 (const Matrix<6, 6> & A);

#endif
