#ifndef QUADRATIC_HEADER
#define QUADRATIC_HEADER

namespace edbdmath
{
   struct quadraticResult_t
   {
      unsigned int numRealSolutions;
      float x[2];
   };

   // Finds any real roots of the quadratic formula:
   //    a * x ^ 2 + b * x + c = 0
   // If there are two real roots, the roots are returned in increasing
   // numerical order.
   quadraticResult_t quadraticRoots(float a, float b, float c);

   float quadraticEval(float a, float b, float c, float x);
}

#endif
