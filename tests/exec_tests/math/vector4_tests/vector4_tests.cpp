#include "vector4.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <iostream>

TEST_CASE("random access operator", "[vector4]")
{
   Vector4 a(0.f, 1.f, 2.f, 3.f);

   REQUIRE( a[0] == 0.f );
   REQUIRE( a[1] == 1.f );
   REQUIRE( a[2] == 2.f );
   REQUIRE( a[3] == 3.f );
}

TEST_CASE("assignment operator", "[vector4]")
{
   Vector4 a(-10.f, -1.f, -2.f, -3.f);

   Vector4 b(0.f, 1.f, 2.f, 3.f);

   b = a;

   REQUIRE( b[0] == -10.f );
   REQUIRE( b[1] == -1.f );
   REQUIRE( b[2] == -2.f );
   REQUIRE( b[3] == -3.f );
}
