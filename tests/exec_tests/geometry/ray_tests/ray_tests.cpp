#include "geometry.hpp"
#include "mesh.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

TEST_CASE( "intersection between ray with sphere-ish mesh", "[ray]" )
{
   geometry::types::triangleMesh_t sphere_mesh = \
      geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::SPHERE);

   Vector3 ray_start(-2.f, -2.f, 0.f);
   Vector3 ray_unit(1.f, 1.f, 0.f);
   ray_unit = ray_unit.Normalize();

   Vector3 intersect_point;
   bool result = geometry::mesh::rayIntersect(sphere_mesh, ray_start, ray_unit, intersect_point);

   REQUIRE(result);
}

TEST_CASE( "non-intersection between ray and sphere-ish mesh", "[ray]" )
{
   geometry::types::triangleMesh_t sphere_mesh = \
      geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::SPHERE);

   Vector3 ray_start(-2.f, -2.f, 0.f);
   Vector3 ray_unit(-1.f, -1.f, 0.f);
   ray_unit = ray_unit.Normalize();

   Vector3 intersect_point;
   bool result = geometry::mesh::rayIntersect(sphere_mesh, ray_start, ray_unit, intersect_point);

   REQUIRE (!result);
}
