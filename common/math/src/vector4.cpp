#include "vector4.hpp"

#include <iostream>

std::ostream & operator << (std::ostream & out, const Vector4 & a)
{
   for (unsigned int j = 0; j < 4; ++j)
   {
      out << std::setw(12) << std::right << std::setprecision(10) << a[j];
      if (j < 3)
      {
         out << ",";
      }
   }
   return out;
}
