#include "plane.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

TEST_CASE( "clip intersecting segment against plane", "[plane]" )
{
   geometry::types::plane_t plane;
   plane.normal[2] = 1.f;
   auto clipped_segment = geometry::plane::clipSegment(
      {0.f, 0.f, 1.f}, {0.f, 0.f, -1.f}, plane
   );

   REQUIRE( clipped_segment.clipped );
}

TEST_CASE( "clip non-intersecting segment against plane", "[plane]" )
{
   geometry::types::plane_t plane;
   plane.normal[2] = 1.f;
   auto clipped_segment = geometry::plane::clipSegment(
      {0.f, 0.f, 1.f}, {0.f, 0.f, 4.f}, plane
   );

   REQUIRE( clipped_segment.clipped );
}

TEST_CASE( "no-clip non-intersecting segment against plane", "[plane]" )
{
   geometry::types::plane_t plane;
   plane.normal[2] = 1.f;
   auto clipped_segment = geometry::plane::clipSegment(
      {0.f, 0.f, -1.f}, {0.f, 0.f, -4.f}, plane
   );

   REQUIRE( !clipped_segment.clipped );
}

TEST_CASE( "clip segment against tilted plane", "[plane]" )
{
   geometry::types::plane_t plane;
   plane.normal[1] = 1.f;
   plane.normal[2] = 1.f;
   auto clipped_segment = geometry::plane::clipSegment(
      {0.f, -2.f, -1.f}, {0.f, 3.f, 4.f}, plane
   );

   REQUIRE( clipped_segment.clipped );
}
