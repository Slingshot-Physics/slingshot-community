#include "quadratic.hpp"

#include <algorithm>
#include <cmath>

namespace edbdmath
{
   quadraticResult_t quadraticRoots(float a, float b, float c)
   {
      quadraticResult_t result;
      const float disc = b * b - 4.f * a * c;
      if (disc < 0.f)
      {
         result.numRealSolutions = 0;
      }
      else if (disc == 0.f)
      {
         result.x[0] = (-1.f * b / (2.f * a));
         if (std::isnan(result.x[0]))
         {
            result.numRealSolutions = 0;
         }
         else
         {
            result.numRealSolutions = 1;
         }
      }
      else
      {
         float u1 = (-1.f * b + std::sqrt(disc)) / (2.f * a);
         float u2 = (-1.f * b - std::sqrt(disc)) / (2.f * a);

         if (std::isnan(u1) && std::isnan(u2))
         {
            result.numRealSolutions = 0;
         }
         else if (!std::isnan(u1) && std::isnan(u2))
         {
            result.x[0] = u1;
            result.numRealSolutions = 1;
         }
         else if (std::isnan(u1) && !std::isnan(u2))
         {
            result.x[0] = u2;
            result.numRealSolutions = 1;
         }
         else
         {
            result.numRealSolutions = 2;
            result.x[0] = std::min(u1, u2);
            result.x[1] = std::max(u1, u2);
         }
      }

      return result;
   }

   float quadraticEval(float a, float b, float c, float x)
   {
      return a * x * x + b * x + c;
   }
}
