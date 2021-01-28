#include <cmath>

#include "data_polygon50.h"
#include "geometry_type_converters.hpp"
#include "geometry_types.hpp"
#include "polygon.hpp"
#include "plane.hpp"
#include "vector3.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#ifndef NUM_SUTHERLAND_HODGMAN_TESTS
#define NUM_SUTHERLAND_HODGMAN_TESTS 25
#endif

// Verifies that the output of this intersection exactly matches a
// pre-computed output.
TEST_CASE( "ground-truth-intersection", "[polygon]")
{
   Vector3 p(0.0f, 0.0f, 0.0f);

   geometry::types::polygon50_t poly1;
   poly1.numVerts = 4;
   poly1.verts[0] = 0.5 * Vector3(1.0f, -1.0f, 0.0f);
   poly1.verts[1] = 0.5 * Vector3(1.0f, 1.0f, 0.0f);
   poly1.verts[2] = 0.5 * Vector3(-1.0f, 1.0f, 0.0f);
   poly1.verts[3] = 0.5 * Vector3(-1.0f, -1.0f, 0.0f);

   geometry::types::polygon50_t poly2;
   poly2.numVerts = 6;
   float a = 1.0f/sqrtf(2.0f);
   poly2.verts[0] = Vector3(0.0f, -0.5f, 0.0f);
   poly2.verts[1] = Vector3(0.4f, -0.5f, 0.0f);
   poly2.verts[2] = Vector3(a, 0.0f, 0.0f);
   poly2.verts[3] = Vector3(0.4f, 0.45f, 0.0f);
   poly2.verts[4] = Vector3(0.0f, a, 0.0f);
   poly2.verts[5] = Vector3(-0.5f, 0.0f, 0.0f);

   geometry::types::polygon50_t output;

   geometry::polygon::convexIntersection(poly1, poly2, output);

   REQUIRE((output.numVerts == 8));
   int idx = 0;
   REQUIRE(output.verts[idx].almostEqual(0.0f, -0.5f, 0.0f));
   ++idx;
   REQUIRE(output.verts[idx].almostEqual(0.4f, -0.5f, 0.0));
   ++idx;
   REQUIRE(output.verts[idx].almostEqual(0.5f, -0.33719018f, 0.0));
   ++idx;
   REQUIRE(output.verts[idx].almostEqual(0.5f, 0.30347116f, 0.0));
   ++idx;
   REQUIRE(output.verts[idx].almostEqual(0.4f, 0.45f, 0.0));
   ++idx;
   REQUIRE(output.verts[idx].almostEqual(0.32221131f, 0.5f, 0.0));
}

// Verifies that all of the vertices from a polygon-vertex-ordering call are
// arranged counter-clockwise.
TEST_CASE( "order polygon vertices CCW", "[polygon]" )
{
   float a = 1.f/sqrtf(2.f);
   geometry::types::polygon50_t unordered_poly;
   unordered_poly.numVerts = 6;
   unordered_poly.verts[0] = Vector3(0.0f, -0.5f, 0.0f);
   unordered_poly.verts[5] = Vector3(0.4f, -0.5f, 0.0f);
   unordered_poly.verts[3] = Vector3(a, 0.0f, 0.0f);
   unordered_poly.verts[1] = Vector3(0.4f, 0.45f, 0.0f);
   unordered_poly.verts[4] = Vector3(0.0f, a, 0.0f);
   unordered_poly.verts[2] = Vector3(-0.5f, 0.0f, 0.0f);

   geometry::types::polygon50_t ordered_poly;
   ordered_poly.numVerts = 6;

   geometry::polygon::orderVertsCcw(
      unordered_poly.verts, unordered_poly.numVerts, ordered_poly.verts
   );

   Vector3 center;
   for (int i = 0; i < ordered_poly.numVerts; ++i)
   {
      center += ordered_poly.verts[i] / (float )(ordered_poly.numVerts);
   }

   for (int i = 0; i < ordered_poly.numVerts; ++i)
   {
      int j = (i + 1) % ordered_poly.numVerts;
      Vector3 cross = (ordered_poly.verts[i] - center).crossProduct(ordered_poly.verts[j] - center);
      REQUIRE(cross[2] >= 0.f);
   }
}

// Verifies that the intersection of two squares with the same dimensions,
// offset from each other, produces another square.
TEST_CASE("square-square intersection", "[polygon]")
{
   geometry::types::polygon50_t poly1;
   poly1.numVerts = 4;
   poly1.verts[0] = Vector3(-0.5f, 0.5f, 0.0f);
   poly1.verts[1] = Vector3(0.5f, 0.5f, 0.0f);
   poly1.verts[2] = Vector3(0.5f, 1.5f, 0.0f);
   poly1.verts[3] = Vector3(-0.5f, 1.5f, 0.0f);

   geometry::types::polygon50_t poly2;
   poly2.numVerts = 4;
   poly2.verts[0] = Vector3(-1.0f, -1.0f, 0.0f);
   poly2.verts[1] = Vector3(1.0f, -1.0f, 0.0f);
   poly2.verts[2] = Vector3(1.0f, 1.0f, 0.0f);
   poly2.verts[3] = Vector3(-1.0f, 1.0f, 0.0f);

   geometry::types::polygon50_t output;

   geometry::polygon::convexIntersection(poly1, poly2, output);

   REQUIRE(output.numVerts == 4);
}

// Generates an arbitrary polygon and checks if the origin is enclosed by the
// polygon.
TEST_CASE("point-inside-polygon", "[polygon]")
{
   Vector3 p(0.0f, 0.0f, 0.0f);

   geometry::types::polygon50_t poly;

   poly.numVerts = 6;
   float a = 1.0f/sqrtf(2.0f);
   poly.verts[0] = Vector3(0.0f, -0.5f, 0.0f);
   poly.verts[1] = Vector3(0.4f, -0.5f, 0.0f);
   poly.verts[2] = Vector3(a, 0.0f, 0.0f);
   poly.verts[3] = Vector3(0.4f, 0.45f, 0.0f);
   poly.verts[4] = Vector3(0.0f, a, 0.0f);
   poly.verts[5] = Vector3(-0.5f, 0.0f, 0.0f);

   REQUIRE(geometry::polygon::pointInside(poly, p));
}

TEST_CASE("full-overlap squares", "[sutherland-hodgman]")
{
   geometry::types::polygon50_t polyA;
   geometry::types::polygon50_t polyAR;
   geometry::types::polygon50_t polyB;

   geometry::types::polygon50_t polyD;

   polyAR.numVerts = 4;
   polyAR.verts[0].Initialize( 1.f,  1.f, 0.f);
   polyAR.verts[1].Initialize( 1.f, -1.f, 0.f);
   polyAR.verts[2].Initialize(-1.f, -1.f, 0.f);
   polyAR.verts[3].Initialize(-1.f,  1.f, 0.f);

   for (unsigned int i = 0; i < NUM_SUTHERLAND_HODGMAN_TESTS; ++i)
   {
      float theta = 2 * M_PI * i / NUM_SUTHERLAND_HODGMAN_TESTS;
      for (unsigned int j = 0; j < polyAR.numVerts; ++j)
      {
         polyA.verts[j].Initialize(
            cosf(theta) * polyAR.verts[j][0] + sinf(theta) * polyAR.verts[j][1],
            -sinf(theta) * polyAR.verts[j][0] + cosf(theta) * polyAR.verts[j][1],
            0.f
         );
      }

      polyA.numVerts = polyAR.numVerts;
      polyB = polyA;

      geometry::polygon::clipSutherlandHodgman(polyA, polyB, polyD);

      REQUIRE( polyD.numVerts == 4 );
   }
}

TEST_CASE("full-overlap pentagons", "[sutherland-hodgman]")
{
   geometry::types::polygon50_t polyA;
   geometry::types::polygon50_t polyAR;
   geometry::types::polygon50_t polyB;

   geometry::types::polygon50_t polyD;

   polyAR.numVerts = 5;
   polyAR.verts[0].Initialize( 1.f,  1.f, 0.f);
   polyAR.verts[1].Initialize( 1.f, -1.f, 0.f);
   polyAR.verts[2].Initialize(-1.f, -1.f, 0.f);
   polyAR.verts[3].Initialize(-1.f,  1.f, 0.f);
   polyAR.verts[4].Initialize(0.f,  1.5f, 0.f);

   for (unsigned int i = 0; i < NUM_SUTHERLAND_HODGMAN_TESTS; ++i)
   {
      float theta = 2 * M_PI * i / NUM_SUTHERLAND_HODGMAN_TESTS;
      for (unsigned int j = 0; j < polyAR.numVerts; ++j)
      {
         polyA.verts[j].Initialize(
            cosf(theta) * polyAR.verts[j][0] + sinf(theta) * polyAR.verts[j][1],
            -sinf(theta) * polyAR.verts[j][0] + cosf(theta) * polyAR.verts[j][1],
            0.f
         );
      }

      polyA.numVerts = polyAR.numVerts;
      polyB = polyA;

      geometry::polygon::clipSutherlandHodgman(polyA, polyB, polyD);

      REQUIRE( polyD.numVerts == 5 );
   }
}

TEST_CASE( "no-op plane projection", "[polygon]")
{
   geometry::types::polygon50_t square;
   square.numVerts = 4;
   square.verts[0].Initialize(-1.f, -1.f, 0.f);
   square.verts[1].Initialize(-1.f, 1.f, 0.f);
   square.verts[2].Initialize(1.f, 1.f, 0.f);
   square.verts[3].Initialize(1.f, -1.f, 0.f);

   geometry::types::polygon50_t projected_polygon = geometry::polygon::projectPolygonToPlane(
      square, {0.f, 0.f, -1.f}, {{0.f, 0.f, 0.f}, {0.f, 0.f, -1.f}}
   );

   REQUIRE( projected_polygon.numVerts == 4 );

   for (unsigned int i = 0; i < projected_polygon.numVerts; ++i)
   {
      REQUIRE( ((projected_polygon.verts[i] - square.verts[i]).magnitude() < 1e-7f) );
   }
}

TEST_CASE( "offset plane projection", "[polygon]")
{
   geometry::types::polygon50_t square;
   square.numVerts = 4;
   square.verts[0].Initialize(-1.f, -1.f, 0.f);
   square.verts[1].Initialize(-1.f, 1.f, 0.f);
   square.verts[2].Initialize(1.f, 1.f, 0.f);
   square.verts[3].Initialize(1.f, -1.f, 0.f);

   geometry::types::polygon50_t projected_polygon = geometry::polygon::projectPolygonToPlane(
      square, {0.f, 0.f, -1.f}, {{0.f, 0.f, 2.f}, {0.f, 0.f, -1.f}}
   );

   REQUIRE( projected_polygon.numVerts == 4 );

   for (unsigned int i = 0; i < projected_polygon.numVerts; ++i)
   {
      REQUIRE( (fabs((projected_polygon.verts[i] - square.verts[i]).magnitude() - 2.f) < 1e-7f) );
   }
}

TEST_CASE( "plane projection", "[polygon]")
{
   geometry::types::polygon50_t square;
   square.numVerts = 4;
   square.verts[0].Initialize(-1.f, -1.f, 0.f);
   square.verts[1].Initialize(-1.f, 1.f, 0.f);
   square.verts[2].Initialize(1.f, 1.f, 0.f);
   square.verts[3].Initialize(1.f, -1.f, 0.f);

   geometry::types::polygon50_t projected_polygon = geometry::polygon::projectPolygonToPlane(
      square, {1.f, 1.f, -1.f}, {{0.f, 0.f, 2.f}, {1.f, 1.f, -1.f}}
   );

   REQUIRE( projected_polygon.numVerts == 4 );

   for (unsigned int i = 0; i < projected_polygon.numVerts; ++i)
   {
      Vector3 projected_vert = geometry::plane::projectPointToPlane(
         {square.verts[i], {1.f, 1.f, -1.f}}, {{0.f, 0.f, 2.f}, {1.f, 1.f, -1.f}}
      );

      REQUIRE( (fabs((projected_polygon.verts[i] - projected_vert).magnitude()) < 1e-7f) );
   }
}
