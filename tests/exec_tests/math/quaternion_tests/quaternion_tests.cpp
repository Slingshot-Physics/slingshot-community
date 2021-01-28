#include "quaternion.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <iostream>

TEST_CASE( "initialization", "[quaternion]" )
{
   Quaternion q(1.f, 0.f, 0.f, 0.f);

   REQUIRE( q[0] == 1.f );
   REQUIRE( q[1] == 0.f );
   REQUIRE( q[2] == 0.f );
   REQUIRE( q[3] == 0.f );
}

TEST_CASE( "const divide by float", "[quaternion]" )
{
   Quaternion q(1.f, 2.f, 3.f, 4.f);
   Quaternion p = q / 6;

   REQUIRE( p[0] == (1.f / 6.f) );
   REQUIRE( p[1] == (2.f / 6.f) );
   REQUIRE( p[2] == (3.f / 6.f) );
   REQUIRE( p[3] == (4.f / 6.f) );
}

TEST_CASE( "divide assign by float", "[quaternion]" )
{
   Quaternion q(1.f, 2.f, 3.f, 4.f);
   q /= 6.f;

   REQUIRE( q[0] == (1.f / 6.f) );
   REQUIRE( q[1] == (2.f / 6.f) );
   REQUIRE( q[2] == (3.f / 6.f) );
   REQUIRE( q[3] == (4.f / 6.f) );
}
