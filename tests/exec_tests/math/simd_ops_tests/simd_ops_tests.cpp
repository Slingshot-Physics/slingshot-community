#include "simd_ops.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <iostream>

#define ARRAY_SIZE 65536

TEST_CASE( "basic simd ops test", "[simd_ops]")
{
   alignas(16) float a[ARRAY_SIZE];
   alignas(16) float b[ARRAY_SIZE];

   // float n = (float )ARRAY_SIZE - 1.f;
   float n = 31;
   float true_val = (ARRAY_SIZE / 32) * (n) * (n + 1.f) * (2.f * n + 1.f) / (3.f * 32.f * 32.f);

   for (unsigned int i = 0; i < ARRAY_SIZE; ++i)
   {
      a[i] = (i % 32) * 1.f / 32.f;
      b[i] = (i % 32) * 2.f / 32.f;
   }

   float val = 0.f;
   int result = dot_product(a, b, ARRAY_SIZE, &val);

   std::cout << "val: " << val << "\n";

   REQUIRE( result == 1 );
   REQUIRE( val > 0.f );
   REQUIRE( val == true_val );
}

TEST_CASE( "simd dot product 4-vectors", "[simd_ops]")
{
   alignas(16) float a[4];
   alignas(16) float b[4];

   for (int i = 0; i < 4; ++i)
   {
      a[i] = 0.f;
      b[i] = 0.f;
   }

   b[2] = -2.f;
   a[2] = 1.f;

   float val = 0.f;
   int result = dot_product(a, b, 4, &val);

   std::cout << "val: " << val << "\n";

   REQUIRE( result == 1 );
   REQUIRE( val == -2.f );
}

TEST_CASE( "simd dot product 8-vectors", "[simd_ops]")
{
   const unsigned int arr_size = 8;
   alignas(16) float a[arr_size];
   alignas(16) float b[arr_size];

   for (int i = 0; i < arr_size; ++i)
   {
      a[i] = 0.f;
      b[i] = 0.f;
   }

   b[2] = -2.f;
   a[2] = 1.f;

   b[6] = -2.f;
   a[6] = 1.f;

   float val = 0.f;
   int result = dot_product(a, b, arr_size, &val);

   std::cout << "val: " << val << "\n";

   REQUIRE( result == 1 );
   REQUIRE( val == -4.f );
}
