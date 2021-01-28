#include "segment.hpp"

#include "attitudeutils.hpp"
#include "random_utils.hpp"
#include "transform.hpp"

#include <iostream>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

int dummy = edbdmath::seed_rng_ret();

// Verify that one pair of non-parallel segments leads to one pair of
// closest points.
TEST_CASE( "1-pair of closest points sanity check", "[segment]" )
{
   Vector3 a(-1.f, 0.f, 1.f);
   Vector3 b(1.f, 0.f, 1.f);

   Vector3 c(0.f, -1.f, -1.f);
   Vector3 d(0.f, 1.f, -1.f);

   const auto closest_points = geometry::segment::closestPointsToSegment(
      a, b, c, d, 1e-7f
   );

   std::cout << "p: " << closest_points.segmentPoints[0] << "\n";
   std::cout << "q: " << closest_points.segmentPoints[1] << "\n";
   std::cout << "r: " << closest_points.otherPoints[0] << "\n";
   std::cout << "s: " << closest_points.otherPoints[1] << "\n";

   REQUIRE(closest_points.numPairs == 1);
}

// Verify that a pair of parallel segments leads to a continuum of closest
// points.
TEST_CASE( "infinite-pair of closest points sanity check", "[segment]" )
{
   Vector3 a(-1.f, 0.f, 1.f);
   Vector3 b(1.f, 0.f, 1.f);

   Vector3 c(-2.f, 0.f, -1.f);
   Vector3 d(2.f, 0.f, -1.f);

   const auto closest_points = geometry::segment::closestPointsToSegment(
      a, b, c, d, 1e-7f
   );

   std::cout << "p: " << closest_points.segmentPoints[0] << "\n";
   std::cout << "q: " << closest_points.segmentPoints[1] << "\n";
   std::cout << "r: " << closest_points.otherPoints[0] << "\n";
   std::cout << "s: " << closest_points.otherPoints[1] << "\n";

   REQUIRE(closest_points.numPairs == 2);
}

// Verify that a pair of parallel segments with partial overlap leads to a
// continuum of closest points.
TEST_CASE( "infinite-pair of closest points sanity check v2", "[segment]" )
{
   Vector3 a(-1.f, 0.f, 1.f);
   Vector3 b(1.f, 0.f, 1.f);

   Vector3 c(-1.5f, 0.f, -2.f);
   Vector3 d(0.35f, 0.f, -2.f);

   const auto closest_points = geometry::segment::closestPointsToSegment(
      a, b, c, d, 1e-7f
   );

   std::cout << "p: " << closest_points.segmentPoints[0] << "\n";
   std::cout << "q: " << closest_points.segmentPoints[1] << "\n";
   std::cout << "r: " << closest_points.otherPoints[0] << "\n";
   std::cout << "s: " << closest_points.otherPoints[1] << "\n";

   REQUIRE(closest_points.numPairs == 2);
}

TEST_CASE( "hyperplane-separated closest points", "[segment-segment]" )
{
   Vector3 a(0.f, 0.f, 0.f);
   Vector3 b(0.f, 0.f, 1.f);

   Vector3 c(0.f, 2.f, 0.5f);
   Vector3 d(0.f, 2.f, 4.f);

   const auto closest_points = geometry::segment::closestPointsToSegment(
      a, b, c, d, 1e-7f
   );

   std::cout << "hyperplane separated test\n";

   std::cout << "p: " << closest_points.segmentPoints[0] << "\n";
   std::cout << "q: " << closest_points.segmentPoints[1] << "\n";
   std::cout << "r: " << closest_points.otherPoints[0] << "\n";
   std::cout << "s: " << closest_points.otherPoints[1] << "\n";

   std::cout << "\n";

   const Vector3 & p = closest_points.segmentPoints[0];
   const Vector3 & q = closest_points.segmentPoints[1];
   const Vector3 & r = closest_points.otherPoints[0];
   const Vector3 & s = closest_points.otherPoints[1];

   REQUIRE( closest_points.numPairs == 2 );
   REQUIRE( geometry::segment::pointColinear(a, b, p) );
   REQUIRE( geometry::segment::pointColinear(a, b, q) );
   REQUIRE( geometry::segment::pointColinear(c, d, r) );
   REQUIRE( geometry::segment::pointColinear(c, d, s) );
}

TEST_CASE( "closest point to point", "[segment]" )
{
   Vector3 a = edbdmath::random_vec3(-32.f, 32.f);
   Vector3 b = edbdmath::random_vec3(-32.f, 32.f);
   Vector3 q = edbdmath::random_vec3(-32.f, 32.f);

   geometry::types::pointBaryCoord_t closest_bary_pt = geometry::segment::closestPointToPoint(
      a, b, q
   );

   Vector3 p = closest_bary_pt.point;
   Vector4 & bary_pt = closest_bary_pt.bary;
   Vector3 p_reconstruct = bary_pt[0] * a + bary_pt[1] * b;

   REQUIRE( (p - p_reconstruct).magnitude() < 1e-3f );
   REQUIRE( bary_pt[0] >= 0.f );
   REQUIRE( bary_pt[1] >= 0.f );
   REQUIRE( bary_pt[0] <= 1.f );
   REQUIRE( bary_pt[1] <= 1.f );
}

TEST_CASE( "degenerate points a,b with intersection", "[segment-segment]")
{
   Vector3 a(-2.f, 2.f, 1.f);
   Vector3 b = a;
   b[0] += 1e-9f;

   Vector3 c(-3.f, 3.f, 2.f);
   Vector3 d(-1.f, 1.f, 0.f);

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 1 );
   REQUIRE( q.magnitude() <= 1e-7f);
   REQUIRE( (p - a).magnitude() < 1e-7f );
}

TEST_CASE( "degenerate points a,b with NO intersection", "[segment-segment]")
{
   Vector3 a(-2.f, 2.f, 1.f);
   Vector3 b = a;
   b[0] += 1e-9f;

   Vector3 c(-3.f, 3.f, 2.f);
   Vector3 d(-1.f, -1.f, 0.f);

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 0 );
   REQUIRE( p.magnitude() <= 1e-7f);
   REQUIRE( q.magnitude() <= 1e-7f);
}

TEST_CASE( "degenerate points c,d with intersection", "[segment-segment]")
{
   Vector3 a(-3.f, 3.f, 2.f);
   Vector3 b(-1.f, 1.f, 0.f);

   Vector3 c(-2.f, 2.f, 1.f);
   Vector3 d = c;
   d[0] += 1e-9f;

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 1 );
   REQUIRE( q.magnitude() <= 1e-7f);
   REQUIRE( (p - c).magnitude() < 1e-7f );
}

TEST_CASE( "degenerate points c,d with NO intersection", "[segment-segment]")
{
   Vector3 a(-3.f, 3.f, 2.f);
   Vector3 b(-2.f, 1.f, 0.f);

   Vector3 c(-2.f, 2.f, 1.f);
   Vector3 d = c;
   d[0] += 1e-9f;

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 0 );
   REQUIRE( p.magnitude() <= 1e-7f);
   REQUIRE( q.magnitude() <= 1e-7f);
}

TEST_CASE( "degenerate points a,b and c,d with intersection", "[segment-segment]")
{
   Vector3 a(-3.f, 3.f, 2.f);
   Vector3 b = a;
   a[2] += 1e-9f;

   Vector3 c = a;
   c[1] += 1e-8f;
   Vector3 d = c;
   d[0] += 1e-9f;

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 1 );
   REQUIRE( (p - a).magnitude() <= 1e-7f);
   REQUIRE( q.magnitude() <= 1e-7f);
}

TEST_CASE( "degenerate points a,b and c,d with NO intersection", "[segment-segment]")
{
   Vector3 a(-3.f, 3.f, 2.f);
   Vector3 b = a;
   a[2] += 1e-9f;

   Vector3 c(2.f, -2.f, 0.f);
   c[1] += 1e-8f;
   Vector3 d = c;
   d[0] += 1e-9f;

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 0 );
   REQUIRE( p.magnitude() <= 1e-7f);
   REQUIRE( q.magnitude() <= 1e-7f);
}

TEST_CASE( "parallel segments intersection", "[segment-segment]" )
{
   Vector3 a(0.f, 1.f, 0.f);
   Vector3 b(-0.5f, 0.f, 0.f);

   Vector3 c(0.5f, 2.f, 0.f);
   Vector3 d(-0.5f, 0.f, 0.f);

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   std::cout << "p: " << p << "\n";
   std::cout << "q: " << q << "\n";

   REQUIRE( num_intersections == 2 );
   REQUIRE( (p - a).magnitude() < 1e-7f );
   REQUIRE( (q - b).magnitude() < 1e-7f );
}

TEST_CASE( "parallel segments intersection2", "[segment-segment]" )
{
   Vector3 a(0.f, 1.f, 0.f);
   Vector3 b(-0.5f, 0.f, 0.f);

   Vector3 c(0.5f, 2.f, 0.f);
   Vector3 d(-1.f, -1.f, 0.f);

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   std::cout << "p: " << p << "\n";
   std::cout << "q: " << q << "\n";

   REQUIRE( num_intersections == 2 );
   REQUIRE( (p - a).magnitude() < 1e-7f );
   REQUIRE( (q - b).magnitude() < 1e-7f );
}

TEST_CASE( "parallel segments NO intersection", "[segment-segment]" )
{
   Vector3 a(0.f, 1.f, 0.f);
   Vector3 b(-0.5f, 0.f, 0.f);

   Vector3 c(1.f, 2.f, 0.f);
   Vector3 d;

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 0 );
}

TEST_CASE( "parallel segments NO intersection y-axis", "[segment-segment]" )
{
   Vector3 a(0.f, 1.f, 0.f);
   Vector3 b(0.f, 0.f, 0.f);

   Vector3 c(0.f, 2.f, 0.f);
   Vector3 d(0.f, 1.5f, 0.f);

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 0 );
}

TEST_CASE( "parallel segments NO intersection x-axis", "[segment-segment]" )
{
   Vector3 a(1.f, 0.f, 0.f);
   Vector3 b(0.f, 0.f, 0.f);

   Vector3 c(2.f, 0.f, 0.f);
   Vector3 d(1.5f, 0.f, 0.f);

   Vector3 p, q;
   int num_intersections = geometry::segment::intersection(
      a, b, c, d, p, q, 1e-7f
   );

   REQUIRE( num_intersections == 0 );
}

// Generates one no-intersection scenario in an initial frame, then randomly
// transforms those points into another rotated/translated frame. Verifies that
// no intersections are detected in the rotated/translated frame.
TEST_CASE( "zero intersection flake test", "[segment-segment]")
{
   Vector3 a(-1.f, -1.f, 0.f);
   Vector3 b(1.f, 1.f, 0.f);

   Vector3 c(-3.f, -3.f, 0.f);
   Vector3 d(-1.f, -2.f, 0.f);

   geometry::types::transform_t trans;
   trans.scale = identityMatrix();
   for (int i = 0; i < 100; ++i)
   {
      float yaw = edbdmath::random_float(-M_PI, M_PI);
      float pitch = edbdmath::random_float(-M_PI/2.f, M_PI/2.f);
      float roll = edbdmath::random_float(-M_PI, M_PI);

      Matrix33 R_frd_to_ned = frd2NedMatrix(roll, pitch, yaw);

      Vector3 translate = edbdmath::random_vec3(-5.f, 5.f);
      trans.rotate = R_frd_to_ned;
      trans.translate = translate;

      Vector3 a_prime = geometry::transform::forwardBound(trans, a);
      Vector3 b_prime = geometry::transform::forwardBound(trans, b);
      Vector3 c_prime = geometry::transform::forwardBound(trans, c);
      Vector3 d_prime = geometry::transform::forwardBound(trans, d);

      Vector3 p, q;
      int num_intersections = geometry::segment::intersection(
         a_prime, b_prime, c_prime, d_prime, p, q, 1e-7f
      );

      REQUIRE( num_intersections == 0 );
   }
}

// Generates a single-intersection scenario in an initial frame, then randomly
// transforms those points into another rotated/translated frame. Verifies that
// one intersection is detected in the rotated/translated frame.
TEST_CASE( "one intersection flake test", "[segment-segment]")
{
   // These segments intersect at the origin.
   Vector3 a(-1.f, -1.f, 1.f);
   Vector3 b(1.f, 1.f, -1.f);

   Vector3 c(1.f, -1.f, -1.f);
   Vector3 d(-1.f, 1.f, 1.f);

   Vector3 p_truth(0.f, 0.f, 0.f);

   geometry::types::transform_t trans;
   trans.scale = identityMatrix();
   for (int i = 0; i < 100; ++i)
   {
      float yaw = edbdmath::random_float(-M_PI, M_PI);
      float pitch = edbdmath::random_float(-M_PI/2.f, M_PI/2.f);
      float roll = edbdmath::random_float(-M_PI, M_PI);

      Matrix33 R_frd_to_ned = frd2NedMatrix(roll, pitch, yaw);

      Vector3 translate = edbdmath::random_vec3(-5.f, 5.f);
      trans.rotate = R_frd_to_ned;
      trans.translate = translate;

      Vector3 a_prime = geometry::transform::forwardBound(trans, a);
      Vector3 b_prime = geometry::transform::forwardBound(trans, b);
      Vector3 c_prime = geometry::transform::forwardBound(trans, c);
      Vector3 d_prime = geometry::transform::forwardBound(trans, d);

      Vector3 p, q;
      int num_intersections = geometry::segment::intersection(
         a_prime, b_prime, c_prime, d_prime, p, q, 1e-7f
      );

      Vector3 p_truth_prime = geometry::transform::forwardBound(trans, p_truth);

      REQUIRE( num_intersections == 1 );
      REQUIRE( (p - p_truth_prime).magnitude() < 1e-6f );
   }
}

// Generates an infinite-intersection scenario in an initial frame, then
// randomly transforms those points into another rotated/translated frame.
// Verifies that infinite intersections are detected in the rotated/translated
// frame.
TEST_CASE( "two intersection flake test", "[segment-segment]" )
{
   Vector3 a(-1.f, -1.f, 0.f);
   Vector3 b(1.f, 1.f, 0.f);

   Vector3 c(0.f, 0.f, 0.f);
   Vector3 d(2.f, 2.f, 0.f);

   Vector3 u_truth(0.f, 0.f, 0.f);
   Vector3 v_truth(1.f, 1.f, 0.f);

   geometry::types::transform_t trans;
   trans.scale = identityMatrix();
   for (int i = 0; i < 100; ++i)
   {
      float yaw = edbdmath::random_float(-M_PI, M_PI);
      float pitch = edbdmath::random_float(-M_PI/2.f, M_PI/2.f);
      float roll = edbdmath::random_float(-M_PI, M_PI);

      Matrix33 R_frd_to_ned = frd2NedMatrix(roll, pitch, yaw);

      Vector3 translate = edbdmath::random_vec3(-5.f, 5.f);
      trans.rotate = R_frd_to_ned;
      trans.translate = translate;

      Vector3 a_prime = geometry::transform::forwardBound(trans, a);
      Vector3 b_prime = geometry::transform::forwardBound(trans, b);
      Vector3 c_prime = geometry::transform::forwardBound(trans, c);
      Vector3 d_prime = geometry::transform::forwardBound(trans, d);

      Vector3 u_truth_prime = geometry::transform::forwardBound(trans, u_truth);
      Vector3 v_truth_prime = geometry::transform::forwardBound(trans, v_truth);

      Vector3 p, q;
      int num_intersections = geometry::segment::intersection(
         a_prime, b_prime, c_prime, d_prime, p, q, 1e-7f
      );

      REQUIRE( num_intersections == 2 );
      if ((u_truth_prime - p).magnitude() < (u_truth_prime - q).magnitude())
      {
         REQUIRE( (u_truth_prime - p).magnitude() < 1e-6f );
         REQUIRE( (v_truth_prime - q).magnitude() < 1e-6f );
      }
      else
      {
         REQUIRE( (u_truth_prime - q).magnitude() < 1e-6f );
         REQUIRE( (v_truth_prime - p).magnitude() < 1e-6f );
      }
   }
}

TEST_CASE( "vertical segment closest points to AABB", "[segment-aabb]" )
{
   Vector3 a(0.f, -2.f, 0.f);
   Vector3 b(0.f, -2.f, 1.f);

   geometry::types::aabb_t aabb;
   aabb.vertMax = {1.f, 1.f, 1.f};
   aabb.vertMin = {-1.f, -1.f, -1.f};

   auto closest_points = geometry::segment::closestPointsToAabb(
      a, b, aabb
   );

   REQUIRE(closest_points.numPairs > 0);
   REQUIRE(
      std::fabs(
         (
            closest_points.segmentPoints[0] - closest_points.otherPoints[0]
         ).magnitude() - 1.f
      ) <= 1e-7f
   );
}

TEST_CASE( "vertical segment touches AABB", "[segment-aabb]" )
{
   Vector3 a(0.f, 0.f, 0.f);
   Vector3 b(0.f, 0.f, 2.f);

   geometry::types::aabb_t aabb;
   aabb.vertMax = {1.f, 1.f, 1.f};
   aabb.vertMin = {-1.f, -1.f, -1.f};

   auto closest_points = geometry::segment::closestPointsToAabb(
      a, b, aabb
   );

   REQUIRE(closest_points.numPairs > 0);
   REQUIRE(
      std::fabs(
         (
            closest_points.segmentPoints[0] - closest_points.otherPoints[0]
         ).magnitude()
      ) <= 1e-7f
   );
}

TEST_CASE( "closest point between degenerate segment and AABB", "[segment-aabb]" )
{
   Vector3 a(-2.f, 0.f, 0.f);
   Vector3 b(-2.f, 0.f, 0.f);

   geometry::types::aabb_t aabb;
   aabb.vertMax = {1.f, 1.f, 1.f};
   aabb.vertMin = {-1.f, -1.f, -1.f};

   auto closest_points = geometry::segment::closestPointsToAabb(
      a, b, aabb
   );

   REQUIRE(closest_points.numPairs == 1);
   REQUIRE(
      std::fabs(
         (
            closest_points.segmentPoints[0] - closest_points.otherPoints[0]
         ).magnitude() - 1.f
      ) <= 1e-7f
   );
}
