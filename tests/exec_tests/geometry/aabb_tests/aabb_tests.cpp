#include "aabb_hull.hpp"
#include "geometry_types.hpp"
#include "support_functions.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

// Verify that the AABB hull for a sphere contains the sphere center and one
// point on the sphere's surface.
TEST_CASE( "points on sphere inside AABB", "[aabb_hull]" )
{
   geometry::types::shapeSphere_t sphere{2.f};
   Vector3 center(1.f, -1.f, 0.5f);
   geometry::types::transform_t trans_C_to_W;
   trans_C_to_W.rotate = identityMatrix();
   trans_C_to_W.scale = identityMatrix();
   trans_C_to_W.translate = center;
   geometry::types::aabb_t sphere_aabb = geometry::aabbHull(trans_C_to_W, sphere);

   Vector3 sphere_point = \
      geometry::supportMapping(Vector3(1.f, 1.f, 1.f), sphere).vert + center;

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE(sphere_aabb.vertMax[i] >= center[i]);
      REQUIRE(sphere_aabb.vertMin[i] <= center[i]);
      REQUIRE(sphere_aabb.vertMax[i] >= sphere_point[i]);
      REQUIRE(sphere_aabb.vertMin[i] <= sphere_point[i]);
   }
}

TEST_CASE( "points on capsule inside AABB", "[aabb_hull]")
{
   geometry::types::shapeCapsule_t cap{3.f, 9.f};
   geometry::types::shapeSphere_t sphere{3.f};
   Vector3 center(1.f, -1.f, 0.5f);
   geometry::types::transform_t trans_C_to_W;
   trans_C_to_W.rotate = identityMatrix();
   trans_C_to_W.scale = identityMatrix();
   trans_C_to_W.translate = center;
   geometry::types::aabb_t cap_aabb = geometry::aabbHull(trans_C_to_W, cap);

   Vector3 cap_point = (
      geometry::supportMapping(Vector3(1.f, 1.f, 1.f), sphere).vert +
      center +
      Vector3(0.f, 0.f, cap.height/2)
   );

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE(cap_aabb.vertMax[i] >= center[i]);
      REQUIRE(cap_aabb.vertMin[i] <= center[i]);
      REQUIRE(cap_aabb.vertMax[i] >= cap_point[i]);
      REQUIRE(cap_aabb.vertMin[i] <= cap_point[i]);
   }
}

// Verify that the AABB hull for a sphere contains the sphere center and one
// point on the sphere's surface.
TEST_CASE( "points on isometric sphere inside AABB", "[aabb_hull]" )
{
   geometry::types::shapeSphere_t sphere{2.f};
   Vector3 center(1.f, -1.f, 0.5f);
   geometry::types::isometricTransform_t trans_C_to_W;
   trans_C_to_W.rotate = identityMatrix();
   trans_C_to_W.translate = center;
   geometry::types::aabb_t sphere_aabb = geometry::aabbHull(trans_C_to_W, sphere);

   Vector3 sphere_point = \
      geometry::supportMapping(Vector3(1.f, 1.f, 1.f), sphere).vert + center;

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE(sphere_aabb.vertMax[i] >= center[i]);
      REQUIRE(sphere_aabb.vertMin[i] <= center[i]);
      REQUIRE(sphere_aabb.vertMax[i] >= sphere_point[i]);
      REQUIRE(sphere_aabb.vertMin[i] <= sphere_point[i]);
   }
}

TEST_CASE( "points on isometric capsule inside AABB", "[aabb_hull]")
{
   geometry::types::shapeCapsule_t cap{3.f, 9.f};
   geometry::types::shapeSphere_t sphere{3.f};
   Vector3 center(1.f, -1.f, 0.5f);
   geometry::types::isometricTransform_t trans_C_to_W;
   trans_C_to_W.rotate = identityMatrix();
   trans_C_to_W.translate = center;
   geometry::types::aabb_t cap_aabb = geometry::aabbHull(trans_C_to_W, cap);

   Vector3 cap_point = (
      geometry::supportMapping(Vector3(1.f, 1.f, 1.f), sphere).vert +
      center +
      Vector3(0.f, 0.f, cap.height/2)
   );

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE(cap_aabb.vertMax[i] >= center[i]);
      REQUIRE(cap_aabb.vertMin[i] <= center[i]);
      REQUIRE(cap_aabb.vertMax[i] >= cap_point[i]);
      REQUIRE(cap_aabb.vertMin[i] <= cap_point[i]);
   }
}
