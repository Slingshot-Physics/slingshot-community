#include "random_utils.hpp"

#include <stdlib.h>
#include <time.h>

namespace edbdmath
{
   void seed_rng(unsigned int seed)
   {
      srand(seed);
   }

   void seed_rng(void)
   {
      srand(time(nullptr));
   }

   int seed_rng_ret(void)
   {
      seed_rng();
      return 1;
   }

   int random_int(int out_min, int out_max)
   {
      int range_len = (out_max - out_min) + 1;
      int mod_range = RAND_MAX - RAND_MAX % range_len;
      int rando = rand();
      while (rando > mod_range)
      {
         rando = rand();
      }
      rando %= range_len;

      return rando + out_min;
   }

   float random_float(void)
   {
      return (float )rand() / RAND_MAX;
   }

   float random_float(float out_min, float out_max)
   {
      return random_float() * (out_max - out_min) + out_min;
   }

   Vector3 random_vec3(float out_min, float out_max)
   {
      Vector3 rando(
         random_float(out_min, out_max),
         random_float(out_min, out_max),
         random_float(out_min, out_max)
      );

      return rando;
   }

   Vector3 random_vec3(const Vector3 & out_min, const Vector3 & out_max)
   {
      Vector3 rando(
         random_float(out_min[0], out_max[0]),
         random_float(out_min[1], out_max[1]),
         random_float(out_min[2], out_max[2])
      );

      return rando;
   }

   Vector3 random_vec3(
      float x_min, float x_max,
      float y_min, float y_max,
      float z_min, float z_max
   )
   {
      Vector3 rando(
         random_float(x_min, x_max),
         random_float(y_min, y_max),
         random_float(z_min, z_max)
      );

      return rando;
   }

   Vector4 random_vec4(float out_min, float out_max)
   {
      Vector4 rando(
         random_float(out_min, out_max),
         random_float(out_min, out_max),
         random_float(out_min, out_max),
         random_float(out_min, out_max)
      );

      return rando;
   }

   Vector4 random_vec4(const Vector4 & out_min, const Vector4 & out_max)
   {
      Vector4 rando(
         random_float(out_min[0], out_max[0]),
         random_float(out_min[1], out_max[1]),
         random_float(out_min[2], out_max[2]),
         random_float(out_min[3], out_max[3])
      );

      return rando;
   }

   Vector4 random_vec4(
      float w_min, float w_max,
      float x_min, float x_max,
      float y_min, float y_max,
      float z_min, float z_max
   )
   {
      Vector4 rando(
         random_float(w_min, w_max),
         random_float(x_min, x_max),
         random_float(y_min, y_max),
         random_float(z_min, z_max)
      );

      return rando;
   }

}
