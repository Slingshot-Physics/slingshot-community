#ifndef SIMD_OPS_HEADER
#define SIMD_OPS_HEADER

// Computes the dot product between two vectors of length 'size' and puts the
// result into the float pointer 'c'. Returns 1 if successful, returns 0 if
// an alignment error is detected.
int dot_product(
   const float * a, const float * b, unsigned int size, float * out
);

// Calculates the dot product between the first 'size' elements of two vectors
// 'a' and 'b'. Does not require 'size' to be evenly divisible by 4.
// Stores the result into the float pointer 'c'. Returns 1 if successful,
// returns 0 if an alignment error is detected.
int masked_dot_product(
   const float * a, const float * b, unsigned int size, float * out
);

int horizontal_add(float * a, float * out);

#endif
