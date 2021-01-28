#ifndef RANDOM_UTILS_HEADER
#define RANDOM_UTILS_HEADER

#include "vector3.hpp"
#include "vector4.hpp"

namespace edbdmath
{
   // Seed the underlying random number generator with a specified seed.
   void seed_rng(unsigned int seed);

   // Seed the underlying random number generator with an unspecified seed.
   void seed_rng(void);

   int seed_rng_ret(void);

   // Returns a random integer in the range [out_min, out_max], inclusive.
   int random_int(int out_min, int out_max);

   // Returns a random float between zero and one.
   float random_float(void);

   // Returns one random float between out_min and out_max.
   float random_float(float out_min, float out_max);

   // Returns a random Vector3 with values between out_min and out_max.
   Vector3 random_vec3(float out_min, float out_max);

   // Returns a random Vector3 with values in each index clamped between
   // out_min and out_max at the corresponding indices.
   Vector3 random_vec3(const Vector3 & out_min, const Vector3 & out_max);

   // Returns a random Vector3 with values in each index clamped between
   // out_min and out_max at the corresponding indices.
   Vector3 random_vec3(
      float x_min, float x_max,
      float y_min, float y_max,
      float z_min, float z_max
   );

   // Returns a random Vector4 with values between out_min and out_max.
   Vector4 random_vec4(float out_min, float out_max);

   // Returns a random Vector4 with values in each index clamped between
   // out_min and out_max at the corresponding indices.
   Vector4 random_vec4(const Vector4 & out_min, const Vector4 & out_max);

   // Returns a random Vector4 with values in each index clamped between
   // out_min and out_max at the corresponding indices.
   Vector4 random_vec4(
      float w_min, float w_max,
      float x_min, float x_max,
      float y_min, float y_max,
      float z_min, float z_max
   );

}

#endif
