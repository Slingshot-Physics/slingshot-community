#include "vector3.hpp"
#include "epa_types.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

// Builds a tetrahedron, deletes a triangle, then re-adds the missing triangle.
TEST_CASE( "tetrahedron is constructed, reduced, and reconstructed properly", "[triangle_grid_mesh]" )
{
   geometry::types::epa::EpaTriangles triangles;

   triangles.add_triangle(0, 1, 3);
   triangles.add_triangle(0, 2, 3);
   triangles.add_triangle(0, 1, 2);
   triangles.add_triangle(1, 2, 3);

   REQUIRE(triangles.size() == 4);

   REQUIRE(triangles[0].neighborIds[0] == 2);
   REQUIRE(triangles[0].neighborIds[1] == 3);
   REQUIRE(triangles[0].neighborIds[2] == 1);

   REQUIRE(triangles[1].neighborIds[0] == 2);
   REQUIRE(triangles[1].neighborIds[1] == 3);
   REQUIRE(triangles[1].neighborIds[2] == 0);

   REQUIRE(triangles[2].neighborIds[0] == 0);
   REQUIRE(triangles[2].neighborIds[1] == 3);
   REQUIRE(triangles[2].neighborIds[2] == 1);

   REQUIRE(triangles[3].neighborIds[0] == 2);
   REQUIRE(triangles[3].neighborIds[1] == 1);
   REQUIRE(triangles[3].neighborIds[2] == 0);

   SECTION( "deleting triangle from tetrahedron leaves all triangles without neighbors" )
   {
      geometry::types::leanNeighborTriangle_t tri = triangles.pop(2);

      REQUIRE(triangles.size() == 3);

      REQUIRE(triangles[0].neighborIds[0] == -1);
      REQUIRE(triangles[0].neighborIds[1] == 2);
      REQUIRE(triangles[0].neighborIds[2] == 1);

      REQUIRE(triangles[1].neighborIds[0] == -1);
      REQUIRE(triangles[1].neighborIds[1] == 2);
      REQUIRE(triangles[1].neighborIds[2] == 0);

      REQUIRE(triangles[2].neighborIds[0] == -1);
      REQUIRE(triangles[2].neighborIds[1] == 1);
      REQUIRE(triangles[2].neighborIds[2] == 0);

      SECTION( "adding deleted triangle makes mesh complete" )
      {
         triangles.add_triangle(0, 1, 2);

         REQUIRE(triangles.size() == 4);

         REQUIRE(triangles[0].neighborIds[0] == 3);
         REQUIRE(triangles[0].neighborIds[1] == 2);
         REQUIRE(triangles[0].neighborIds[2] == 1);

         REQUIRE(triangles[1].neighborIds[0] == 3);
         REQUIRE(triangles[1].neighborIds[1] == 2);
         REQUIRE(triangles[1].neighborIds[2] == 0);

         REQUIRE(triangles[2].neighborIds[0] == 3);
         REQUIRE(triangles[2].neighborIds[1] == 1);
         REQUIRE(triangles[2].neighborIds[2] == 0);

         REQUIRE(triangles[3].neighborIds[0] == 0);
         REQUIRE(triangles[3].neighborIds[1] == 2);
         REQUIRE(triangles[3].neighborIds[2] == 1);
      }
   }
}
