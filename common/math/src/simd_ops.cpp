#include "simd_ops.hpp"

#include <emmintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#include <cstddef>

#include <iostream>

int dot_product(const float * a, const float * b, unsigned int size, float * out)
{
   if (
      ((std::size_t )a % 16 != 0) ||
      ((std::size_t )b % 16 != 0) ||
      (size % 4 != 0)
   )
   {
      return 0;
   }

   __m128 s_vec_a;
   __m128 s_vec_b;
   __m128 s_hadamard;
   __m128 s_sum;

   s_sum = _mm_set_ps1(0.f);

   unsigned int num_loops = size / 4;

   for (unsigned int i = 0; i < num_loops; ++i)
   {
      // Load the next 16 bytes of data into the registers.
      s_vec_a = _mm_load_ps(a + (i * 4));
      s_vec_b = _mm_load_ps(b + (i * 4));

      // Calculate the hadamard product between four floats per input.
      s_hadamard = _mm_mul_ps(s_vec_a, s_vec_b);

      // Sum up elements of each register.
      s_sum = _mm_add_ps(s_hadamard, s_sum);
   }

   // 's_sum' ends up with four elements that need to be added together.
   // This performs a horizontal add between all of the elements in 's_sum'.
   // Pilfered from:
   //    https://bit.ly/3jAdO7d
   __m128 s_shuf = _mm_shuffle_ps(s_sum, s_sum, _MM_SHUFFLE(2, 3, 0, 1));
   __m128 s_fsum = _mm_add_ps(s_sum, s_shuf);
   s_shuf = _mm_movehl_ps(s_shuf, s_fsum);
   s_fsum = _mm_add_ss(s_fsum, s_shuf);
   *out = _mm_cvtss_f32(s_fsum);

   // This does the exact same thing as the code above, it's just less
   // portable.
   // s_sum = _mm_hadd_ps(s_sum, s_sum);
   // s_sum = _mm_hadd_ps(s_sum, s_sum);
   // *out = _mm_cvtss_f32(s_sum);

   return 1;
}

int masked_dot_product(
   const float * a, const float * b, unsigned int size, float * out
)
{
   // Note that there's no requirement for a and b to be on 16-byte boundaries.
   unsigned int padded_size = size + (size % 4 != 0) * (4 - size % 4);
   unsigned int num_loops = padded_size / 4;

   __m128 s_sum = _mm_set1_ps(0.f);
   __m128 s_vec_a;
   __m128 s_vec_b;
   __m128 s_hadamard;

   for (unsigned int i = 0; i < num_loops; ++i)
   {
      s_vec_a = _mm_set_ps(
         (i + 0 < size) ? a[i + 0] : 0.f,
         (i + 1 < size) ? a[i + 1] : 0.f,
         (i + 2 < size) ? a[i + 2] : 0.f,
         (i + 3 < size) ? a[i + 3] : 0.f
      );
      s_vec_b = _mm_set_ps(
         (i + 0 < size) ? b[i + 0] : 0.f,
         (i + 1 < size) ? b[i + 1] : 0.f,
         (i + 2 < size) ? b[i + 2] : 0.f,
         (i + 3 < size) ? b[i + 3] : 0.f
      );
      s_hadamard = _mm_mul_ps(s_vec_a, s_vec_b);
      s_sum = _mm_add_ps(s_hadamard, s_sum);
   }

   __m128 s_shuf = _mm_shuffle_ps(s_sum, s_sum, _MM_SHUFFLE(2, 3, 0, 1));
   __m128 s_fsum = _mm_add_ps(s_sum, s_shuf);
   s_shuf = _mm_movehl_ps(s_shuf, s_fsum);
   s_fsum = _mm_add_ss(s_fsum, s_shuf);
   *out = _mm_cvtss_f32(s_fsum);

   return 1;
}
