#include "data_model_io.h"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <stdio.h>

TEST_CASE( "vec3 --> file --> vec3" )
{
   data_vector3_t temp_vec;
   temp_vec.v[0] = -1.f;
   temp_vec.v[1] = 21.24356f;
   temp_vec.v[2] = -17.2f;

   int write_result = write_data_to_file(&temp_vec, "data_vector3_t", "temp_vec3.json");

   REQUIRE(write_result != 0);

   data_vector3_t temp_vec_from_file;

   int read_result = read_data_from_file(&temp_vec_from_file, "temp_vec3.json");

   REQUIRE(read_result != 0);

   for (unsigned int i = 0; i < 3; ++i)
   {
      REQUIRE( fabs(temp_vec.v[i] - temp_vec_from_file.v[i]) < 1e-6f);
   }
}
