#include "vector3.hpp"

#include <iostream>

std::ostream & operator << (std::ostream & out, const Vector3 & a)
{
   for (unsigned int j = 0; j < 3; ++j)
   {
      out << std::setw(12) << std::right << std::setprecision(10) << a[j];
      if (j < 2)
      {
         out << ",";
      }
   }
   return out;
}

Vector3 operator* (const float & v, const Vector3 & a)
{
   return a * v;
}
