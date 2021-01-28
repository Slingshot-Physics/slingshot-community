#include "line.hpp"

#include <iostream>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

TEST_CASE( "1-pair of closest points sanity check", "[line]" )
{
   Vector3 a(-1.f, 0.f, 1.f);
   Vector3 b(1.f, 0.f, 1.f);

   Vector3 c(0.f, -1.f, -1.f);
   Vector3 d(0.f, 1.f, -1.f);

   Vector3 p, q;

   int num_pairs = geometry::line::closestPointsToLine(
      a, b, c, d, p, q, 1e-7f
   );

   std::cout << "p: " << p << "\n";
   std::cout << "q: " << q << "\n";

   REQUIRE(num_pairs == 1);
}

TEST_CASE( "infinite-pair of closest points sanity check", "[line]" )
{
   Vector3 a(-1.f, 0.f, 1.f);
   Vector3 b(1.f, 0.f, 1.f);

   Vector3 c(-2.f, 0.f, -1.f);
   Vector3 d(2.f, 0.f, -1.f);

   Vector3 p, q;

   int num_pairs = geometry::line::closestPointsToLine(
      a, b, c, d, p, q, 1e-7f
   );

   std::cout << "p: " << p << "\n";
   std::cout << "q: " << q << "\n";

   REQUIRE(num_pairs == 2);
}

TEST_CASE( "parallel, non-intersecting line and segment", "[line]" )
{
   Vector3 a(0.0f, 0.f, 0.f);
   Vector3 b(1.f, 1.f, 0.f);

   Vector3 c(0.f, 2.f, 0.f);
   Vector3 d(0.5f, 1.5f, 0.f);

   Vector3 p, q;
   int result = geometry::line::segmentIntersection(a, b, c, d, p, q);

   // Might consider letting the algorithm set p and q to nan in this case.
   REQUIRE(result == 0);
}

TEST_CASE( "single intersecting line and segment", "[line]" )
{
   Vector3 a(0.0f, 0.f, 0.f);
   Vector3 b(1.f, 1.f, 0.f);

   Vector3 c(0.f, 2.f, 0.f);
   Vector3 d(2.5f, -1.5f, 0.f);

   Vector3 p, q;
   int result = geometry::line::segmentIntersection(a, b, c, d, p, q);

   REQUIRE(result == 1);
}

TEST_CASE( "infinitely intersecting line and segment", "[line]" )
{
   Vector3 a(0.0f, 0.f, 0.f);
   Vector3 b(1.f, 1.f, 0.f);

   Vector3 c(2.f, 2.f, 0.f);
   Vector3 d(2.5f, 2.5f, 0.f);

   Vector3 p, q;
   int result = geometry::line::segmentIntersection(a, b, c, d, p, q);

   // In the case of an infinite number of intersection points, the points of
   // intersection will be the start and end points of the line segment.
   REQUIRE(p == c);
   REQUIRE(q == d);
   REQUIRE(result == 2);
}

// Generates a series of tests where a line segment is spun in a yz-constrained
// circle, while its center is moved to either side of a vertical (z-hat) line.
// A line segment should intersect the line in some, but not all, cases.
TEST_CASE( "spinning segment intersection test", "[line]" )
{
   float seg_length = 10.f;
   Vector3 line_start(0.f, 0.f, -1.f);
   Vector3 line_end(0.f, 0.f, 1.f);
   for (int i = 0; i < 100; ++i)
   {
      float y = (float )(i - 50)/100.f;

      Vector3 segment_start(0.f, y, 0.f);
      int num_segments = 100;

      for (int j = 0; j < num_segments; ++j)
      {
         float theta = ((float )j/(float )num_segments) * 2 * M_PI - M_PI;
         Vector3 segment_slope(
            0.f, cosf(theta), sinf(theta)
         );

         Vector3 segment_end = segment_start + segment_slope * seg_length;

         Vector3 p, q;
         int result = geometry::line::segmentIntersection(
            line_start, line_end, segment_start, segment_end, p, q
         );

         Vector3 zero;

         float big_dot = (seg_length * segment_slope.dot(zero - segment_start));

         bool desired_result = (
            (
               ((big_dot <= seg_length) && (big_dot >= 0) && result >= 1)
               || ((big_dot >= 0) && (big_dot < seg_length) && result == 0)
               || ((big_dot < 0) && result == 0)
            )
         );
         REQUIRE(desired_result);
      }
   }
}

