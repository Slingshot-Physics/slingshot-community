#include "matrix33.hpp"

#include <iostream>

std::ostream & operator << (std::ostream & out, const Matrix33 & M)
{
   for (unsigned int i = 0; i < 3; ++i)
   {
      for (unsigned int j = 0; j < 3; ++j)
      {
         out << std::setw(10) << std::right << M(i, j);
         if (j < 2)
         {
            out << " ";
         }
         else
         {
            out << "\n";
         }
      }
   }
   return out;
}

Matrix33 crossProductMatrix(const Vector3 & a)
{
   Matrix33 A(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
   A(0, 1) = -1*a[2];
   A(0, 2) = a[1];
   A(1, 2) = -1*a[0];
   A(1, 0) = a[2];
   A(2, 0) = -1*a[1];
   A(2, 1) = a[0];
   return A;
}

Matrix33 identityMatrix(void)
{
   Matrix33 I(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
   return I;
}

Matrix33 outerProduct(const Vector3 & a, const Vector3 & b)
{
   Matrix33 A(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
   for (unsigned int i = 0; i < 3; ++i)
   {
      for (unsigned int j = 0; j < 3; ++j)
      {
         A(i, j) = a[i] * b[j];
      }
   }
   return A;
}

Matrix33 operator* (const float & v, const Matrix33 & M)
{
   return M*v;
}
