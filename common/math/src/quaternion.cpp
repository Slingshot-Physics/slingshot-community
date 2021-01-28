#include "quaternion.hpp"

#include <iostream>

Quaternion operator* (const float & a, const Quaternion b)
{
   Quaternion result(b);
   return result*a;
}

Quaternion exp (const Vector3 & u)
{
   Quaternion result;
   result.scalar() = cos(u.magnitude());
   result.vector() = sin(u.magnitude())*u.unitVector();
   result = result/result.magnitude();
   return result;
}

Quaternion exp (const Quaternion & u)
{
   return exp(u.vector());
}

std::ostream & operator << (std::ostream & out, const Quaternion & a)
{
   for (unsigned int j = 0; j < 4; ++j)
   {
      out << std::setw(10) << std::right << a[j];
      if (j < 3)
      {
         out << " ";
      }
      else
      {
         //out << "\n";
      }
   }
   return out;
}
