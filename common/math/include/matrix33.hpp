#ifndef MATRIX33
#define MATRIX33

#include <cstring>
#include <iostream>
#include <iomanip>

#include "vector3.hpp"

class Matrix33
{
   public:

      Matrix33 (
         float a00, float a01, float a02,
         float a10, float a11, float a12,
         float a20, float a21, float a22
      )
      {
         Initialize (
            a00, a01, a02,
            a10, a11, a12,
            a20, a21, a22
         );
      }

      Matrix33 (const Matrix33 & M)
      {
         for (int i = 0; i < 9; ++i)
         {
            base_[i] = M.base_[i];
         }
      }

      Matrix33 (void)
      {
         Initialize (
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
         );
      }

      static Matrix33 diag (const Vector3 & v)
      {
         return Matrix33(
            v[0], 0.f, 0.f,
            0.f, v[1], 0.f,
            0.f, 0.f, v[2]
         );
      }

      static Matrix33 diagInv (const Vector3 & v)
      {
         return Matrix33(
            1.f / v[0], 0.f, 0.f,
            0.f, 1.f / v[1], 0.f,
            0.f, 0.f, 1.f / v[2]
         );
      }

      void Initialize (
         float a00, float a01, float a02,
         float a10, float a11, float a12,
         float a20, float a21, float a22
      )
      {
         base_[index(0, 0)] = a00; base_[index(0, 1)] = a01; base_[index(0, 2)] = a02;
         base_[index(1, 0)] = a10; base_[index(1, 1)] = a11; base_[index(1, 2)] = a12;
         base_[index(2, 0)] = a20; base_[index(2, 1)] = a21; base_[index(2, 2)] = a22;
      }

      Matrix33 diagInv(void) const
      {
         return Matrix33(
            1.f / (*this)(0, 0), 0.f, 0.f,
            0.f, 1.f / (*this)(1, 1), 0.f,
            0.f, 0.f, 1.f / (*this)(2, 2)
         );
      }

      Matrix33 transpose (void) const
      {
         Matrix33 T;
         for (unsigned int i = 0; i < 3; ++i)
         {
            for (unsigned int j = 0; j < 3; ++j)
            {
               T.base_[T.index(i, j)] = base_[index(j, i)];
            }
         }

         return T;
      }

      Matrix33 & transposeInPlace (void)
      {
         Matrix33 T;
         for (unsigned int i = 0; i < 3; ++i)
         {
            for (unsigned int j = 0; j < 3; ++j)
            {
               T(i, j) = base_[index(j, i)];
            }
         }

         std::memcpy(base_, T.base_, sizeof(float) * 9);

         return *this;
      }

      float determinant (void) const
      {
         float det = \
            base_[index(0, 0)]*base_[index(1, 1)]*base_[index(2, 2)] +
            base_[index(0, 1)]*base_[index(1, 2)]*base_[index(2, 0)] +
            base_[index(0, 2)]*base_[index(1, 0)]*base_[index(2, 1)] -
            base_[index(2, 0)]*base_[index(1, 1)]*base_[index(0, 2)] -
            base_[index(2, 1)]*base_[index(1, 2)]*base_[index(0, 0)] -
            base_[index(2, 2)]*base_[index(1, 0)]*base_[index(0, 1)];
         return det;
      }

      Matrix33 & operator= (const Matrix33 & M)
      {
         if (this == &M)
         {
            return *this;
         }

         std::memcpy(base_, M.base_, sizeof(float) * 9);
         return *this;
      }

      const Matrix33 operator= (const Matrix33 & M) const
      {
         if (&M == this)
         {
            return *this;
         }

         Matrix33 result(*this);
         return result;
      }

      float & operator() (unsigned int row, unsigned int col)
      {
         return base_[index(row, col)];
      }

      float operator() (unsigned int row, unsigned int col) const
      {
         return base_[index(row, col)];
      }

      Matrix33 operator* (const float & v) const
      {
         Matrix33 result;
         for (unsigned int i = 0; i < 9; ++i)
         {
            result.base_[i] = base_[i] * v;
         }
         return result;
      }

      Vector3 operator* (const Vector3 & a) const
      {
         Vector3 result;
         for (unsigned int i = 0; i < 3; ++i)
         {
            for (unsigned int j = 0; j < 3; ++j)
            {
               result[i] += base_[index(i, j)]*a[j];
            }
         }
         return result;
      }

      Matrix33 operator* (const Matrix33 & M) const
      {
         Matrix33 result;
         for (unsigned int i = 0; i < 3; ++i)
         {
            for (unsigned int j = 0; j < 3; ++j)
            {
               for (unsigned int k = 0; k < 3; ++k)
               {
                  result(i, j) += base_[index(i, k)] * M(k, j);
               }
            }
         }

         return result;
      }

      Matrix33 & operator*= (const float & v)
      {
         for (unsigned int i = 0; i < 9; ++i)
         {
            base_[i] *= v;
         }

         return *this;
      }

      Matrix33 & operator*= (const Matrix33 & M)
      {
         Matrix33 result;

         for (unsigned int i = 0; i < 3; ++i)
         {
            for (unsigned int j = 0; j < 3; ++j)
            {
               for (unsigned int k = 0; k < 3; ++k)
               {
                  result(i, j) += base_[index(i, k)] * M(k, j);
               }
            }
         }

         std::memcpy(base_, result.base_, sizeof(float) * 9);

         return *this;
      }

      Matrix33 operator/ (const float & v) const
      {
         Matrix33 A(*this);

         for (unsigned int i = 0; i < 9; ++i)
         {
            A.base_[i] /= v;
         }

         return A;
      }

      Matrix33 & operator/= (const float & v)
      {
         for (unsigned int i = 0; i < 9; ++i)
         {
            base_[i] /= v;
         }

         return *this;
      }

      Matrix33 operator+ (const Matrix33 & M) const
      {
         Matrix33 result;
         for (unsigned int i = 0; i < 9; ++i)
         {
            result.base_[i] = base_[i] + M.base_[i];
         }

         return result;
      }

      Matrix33 & operator+= (const Matrix33 & M)
      {
         for (unsigned int i = 0; i < 9; ++i)
         {
            base_[i] += M.base_[i];
         }

         return *this;
      }

      Matrix33 operator- (const Matrix33 & M) const
      {
         Matrix33 result;
         for (unsigned int i = 0; i < 9; ++i)
         {
            result.base_[i] = base_[i] - M.base_[i];
         }

         return result;
      }

      Matrix33 & operator-= (const Matrix33 & M)
      {
         for (unsigned int i = 0; i < 9; ++i)
         {
            base_[i] -= M.base_[i];
         }

         return *this;
      }

      Matrix33 operator~ (void) const
      {
         Matrix33 A(*this);
         float det = determinant();
         // |00 01 02|
         // |10 11 12|
         // |20 21 22|
         A.base_[index(0, 0)] =       base_[index(1, 1)]*base_[index(2, 2)] - base_[index(1, 2)]*base_[index(2, 1)];
         A.base_[index(1, 0)] = -1.0*(base_[index(1, 0)]*base_[index(2, 2)] - base_[index(1, 2)]*base_[index(2, 0)]);
         A.base_[index(2, 0)] =       base_[index(1, 0)]*base_[index(2, 1)] - base_[index(1, 1)]*base_[index(2, 0)];
         A.base_[index(0, 1)] = -1.0*(base_[index(0, 1)]*base_[index(2, 2)] - base_[index(0, 2)]*base_[index(2, 1)]);
         A.base_[index(1, 1)] =       base_[index(0, 0)]*base_[index(2, 2)] - base_[index(0, 2)]*base_[index(2, 0)];
         A.base_[index(2, 1)] = -1.0*(base_[index(0, 0)]*base_[index(2, 1)] - base_[index(0, 1)]*base_[index(2, 0)]);
         A.base_[index(0, 2)] =       base_[index(0, 1)]*base_[index(1, 2)] - base_[index(0, 2)]*base_[index(1, 1)];
         A.base_[index(1, 2)] = -1.0*(base_[index(0, 0)]*base_[index(1, 2)] - base_[index(0, 2)]*base_[index(1, 0)]);
         A.base_[index(2, 2)] =       base_[index(0, 0)]*base_[index(1, 1)] - base_[index(0, 1)]*base_[index(1, 0)];

         A /= det;
         return A;
      }

   private:

      unsigned int index(unsigned int row, unsigned int col) const
      {
         return 3 * row + col;
      }

      float base_[9];

};

std::ostream & operator << (std::ostream & out, const Matrix33 & M);

Matrix33 crossProductMatrix(const Vector3 & a);

Matrix33 identityMatrix(void);

Matrix33 outerProduct(const Vector3 & a, const Vector3 & b);

Matrix33 operator* (const float & v, const Matrix33 & M);

#endif
