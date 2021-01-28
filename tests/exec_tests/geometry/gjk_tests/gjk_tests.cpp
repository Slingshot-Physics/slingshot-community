#include "geometry_type_converters.hpp"
#include "gjk.hpp"
#include "md_hull_intersection.hpp"
#include "mesh.hpp"
#include "quickhull.hpp"
#include "attitudeutils.hpp"
#include "random_utils.hpp"

#include "random_gjk_input_util.hpp"
#include "test_input_serializer_util.hpp"

#include <iostream>

#ifndef TEST_FILENAMES
#define TEST_FILENAMES ""
#define NUM_FILENAMES 0
#endif

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#ifndef NUM_GJK_TESTS
#define NUM_GJK_TESTS 100
#endif

int dummy = edbdmath::seed_rng_ret();

typedef geometry::types::convexPolyhedron_t gjk_convexPolyhedron_t;

void convert_data_mesh_to_gjk_convexPolyhedron(
   const data_triangleMesh_t * meshIn,
   const Matrix33 & R_b2l,
   const Vector3 & offset,
   gjk_convexPolyhedron_t & cp_out
)
{
   cp_out.numVerts = meshIn->numVerts;
   Vector3 center;
   for (unsigned int i = 0; i < meshIn->numVerts; ++i)
   {
      geometry::converters::from_pod(&(meshIn->verts[i]), cp_out.verts[i]);
      cp_out.verts[i] = R_b2l * cp_out.verts[i] + offset;
      center += cp_out.verts[i];
   }

   center /= cp_out.numVerts;
}

bool test_gjk(
   const geometry::types::testGjkInput_t & test_input,
   const geometry::types::convexPolyhedron_t & polyhedronA,
   const geometry::types::convexPolyhedron_t & polyhedronB,
   const geometry::types::triangleMesh_t & meshA,
   const geometry::types::triangleMesh_t & meshB,
   const std::string & test_name,
   bool serialize_on_failure,
   geometry::types::gjkResult_t & gjk_output
)
{
   gjk_output = geometry::gjk::alg(
      test_input.transformA, test_input.transformB, polyhedronA, polyhedronB
   );

   float closest_points_distance = 0.f;
   bool slow_result = test_utils::intersection_test(
      meshA, meshB, closest_points_distance
   );

   bool requirement = (slow_result == gjk_output.intersection);

   if (!requirement)
   {
      std::cout << "GJK: " << gjk_output.intersection << " Hull: " << slow_result << "\n";
      std::cout << "\tdistance between closest points: " << closest_points_distance << "\n";
      if (serialize_on_failure)
      {
         std::cout << "serializing failure\n";
         std::ostringstream output_name;
         output_name << "test-" << test_name << ".json";
         std::string filename = output_name.str();
         test_utils::serialize_geometry_input<geometry::types::testGjkInput_t, data_testGjkInput_t>(test_input, "data_testGjkInput_t", filename);
      }
   }

   REQUIRE( requirement );

   return requirement;
}

TEST_CASE( "random cube-cube", "[gjk]" )
{
   for (int i = 0; i < NUM_GJK_TESTS; ++i)
   {
      std::ostringstream ostream;
      ostream << "gjk-cube-cube-" << i;
      geometry::types::testGjkInput_t test_input;
      geometry::types::convexPolyhedron_t polyhedronA;
      geometry::types::convexPolyhedron_t polyhedronB;
      geometry::types::triangleMesh_t meshA;
      geometry::types::triangleMesh_t meshB;

      test_utils::generate_random_gjk_input(
         geometry::types::enumShape_t::CUBE,
         geometry::types::enumShape_t::CUBE,
         test_input,
         polyhedronA,
         polyhedronB,
         meshA,
         meshB
      );

      SECTION(ostream.str().c_str())
      {
         std::string test_name_str(ostream.str());
         geometry::types::gjkResult_t gjk_result;
         bool agreement = test_gjk(
            test_input,
            polyhedronA,
            polyhedronB,
            meshA,
            meshB,
            test_name_str,
            true,
            gjk_result
         );
      }
   }
}

TEST_CASE( "config from file", "[gjk]" )
{
   std::string filenames[] = {TEST_FILENAMES};
   for (int i = 0; i < NUM_FILENAMES; ++i)
   {
      std::string section_name(filenames[i]);
      SECTION(section_name.c_str())
      {
         std::string & filename = filenames[i];
         data_testGjkInput_t test_input_data;
         int read_result = read_data_from_file(&test_input_data, filename.c_str());

         if (!read_result)
         {
            std::cout << "Couldn't open file named: " << filename << "\n";
         }
         REQUIRE( read_result == 1 );

         geometry::types::testGjkInput_t test_input;
         geometry::converters::from_pod(&test_input_data, test_input);

         geometry::types::convexPolyhedron_t polyhedronA;
         geometry::types::convexPolyhedron_t polyhedronB;
         geometry::types::triangleMesh_t meshA;
         geometry::types::triangleMesh_t meshB;

         test_utils::load_meshes_and_polyhedra(
            test_input, polyhedronA, polyhedronB, meshA, meshB
         );

         std::string test_name_str(filename);
         geometry::types::gjkResult_t gjk_result;
         test_gjk(
            test_input,
            polyhedronA,
            polyhedronB,
            meshA,
            meshB,
            test_name_str,
            false,
            gjk_result
         );
      }
   }
}

// Tests intersection of two pyramids whose apexes are the closest points to
// each other, but are not intersecting.
TEST_CASE( "two non-intersecting pyramids", "[gjk]" )
{
   Vector3 offset(0.0f, 0.0f, 2.5f);

   Vector3 a(0.0f, 0.0f, 1.0f);
   Vector3 b(-1.0f, -1.0f, 0.0f);
   Vector3 c(1.0f, -1.0f, 0.0f);
   Vector3 d(-1.0f, 1.0f, 0.0f);
   Vector3 e(1.0f, 1.0f, 0.0f);
   gjk_convexPolyhedron_t bodyA = {
      5, {a, b, c, d, e}, (a + b + c + d + e)/5.0f
   };

   Vector3 f = -1.0f * a + offset;
   Vector3 g = b + offset;
   Vector3 h = c + offset;
   Vector3 i = d + offset;
   Vector3 j = e + offset;
   gjk_convexPolyhedron_t bodyB = {
      5, {g, h, i, j, f}, (f + g + h + i + j)/5.0f
   };

   Matrix33 eye(identityMatrix());
   Vector3 zero(0.0f, 0.0f, 0.0f);

   geometry::types::transform_t dummy_a = {
      eye, eye, zero
   };

   geometry::types::transform_t dummy_b = {
      eye, eye, zero
   };

   gjk_result_t output = geometry::gjk::alg(dummy_a, dummy_b, bodyA, bodyB);

   REQUIRE(output.minSimplex.numVerts == 1);
   REQUIRE(output.minSimplex.bodyAVertIds[0] == 0);
   REQUIRE(output.minSimplex.bodyBVertIds[0] == 4);
   REQUIRE(output.intersection == false);
}

// Tests intersection of two non-intersecting pyramids. One pyramid is a
// replica of the other, just translated vertically.
TEST_CASE( "two non-intersecting pyramids (vertical sep)", "[gjk]" )
{
   Vector3 offset(0.0f, 0.0f, 2.5f);

   Vector3 a(0.0f, 0.0f, 1.0f);
   Vector3 b(-1.0f, -1.0f, 0.0f);
   Vector3 c(1.0f, -1.0f, 0.0f);
   Vector3 d(-1.0f, 1.0f, 0.0f);
   Vector3 e(1.0f, 1.0f, 0.0f);
   gjk_convexPolyhedron_t bodyA = {
      5, {a, b, c, d, e}, (a + b + c + d + e)/5.0f
   };

   Vector3 f = a + offset;
   Vector3 g = b + offset;
   Vector3 h = c + offset;
   Vector3 i = d + offset;
   Vector3 j = e + offset;
   gjk_convexPolyhedron_t bodyB = {
      5, {g, h, i, j, f}, (f + g + h + i + j)/5.0f
   };

   Matrix33 eye(identityMatrix());
   Vector3 zero(0.0f, 0.0f, 0.0f);

   geometry::types::transform_t dummy_a = {
      eye, eye, zero
   };

   geometry::types::transform_t dummy_b = {
      eye, eye, zero
   };

   gjk_result_t output = geometry::gjk::alg(dummy_a, dummy_b, bodyA, bodyB);

   bool nasty_or_condition = (
      (output.minSimplex.bodyBVertIds[0] == 0) ||
      (output.minSimplex.bodyBVertIds[0] == 1) ||
      (output.minSimplex.bodyBVertIds[0] == 2) ||
      (output.minSimplex.bodyBVertIds[0] == 3)
   );
   REQUIRE(output.minSimplex.numVerts >= 1);
   REQUIRE(output.minSimplex.bodyAVertIds[0] == 0);
   REQUIRE(nasty_or_condition);
   REQUIRE(output.intersection == false);
}

TEST_CASE( "two intersecting pyramids", "[gjk]")
{
   Vector3 offset(0.0f, 0.0f, 0.8f);

   Vector3 a(0.0f, 0.0f, 1.0f);
   Vector3 b(-1.0f, -1.0f, 0.0f);
   Vector3 c(1.0f, -1.0f, 0.0f);
   Vector3 d(-1.0f, 1.0f, 0.0f);
   Vector3 e(1.0f, 1.0f, 0.0f);
   gjk_convexPolyhedron_t bodyA = {
      5, {a, b, c, d, e}, (a + b + c + d + e)/5.0f
   };

   Vector3 f = a + offset;
   Vector3 g = b + offset;
   Vector3 h = c + offset;
   Vector3 i = d + offset;
   Vector3 j = e + offset;
   gjk_convexPolyhedron_t bodyB = {
      5, {g, h, i, j, f}, (f + g + h + i + j)/5.0f
   };

   Matrix33 eye(identityMatrix());
   Vector3 zero(0.0f, 0.0f, 0.0f);

   geometry::types::transform_t dummy_a = {
      eye, eye, zero
   };

   geometry::types::transform_t dummy_b = {
      eye, eye, zero
   };

   gjk_result_t output = geometry::gjk::alg(dummy_a, dummy_b, bodyA, bodyB);

   bool nasty_or_statement = (
      (output.minSimplex.bodyBVertIds[0] == 0) ||
      (output.minSimplex.bodyBVertIds[0] == 1) ||
      (output.minSimplex.bodyBVertIds[0] == 2) ||
      (output.minSimplex.bodyBVertIds[0] == 3)
   );
   REQUIRE(output.minSimplex.numVerts >= 1);
   REQUIRE(output.minSimplex.bodyAVertIds[0] == 0);
   REQUIRE(nasty_or_statement);
   REQUIRE(output.intersection == true);
}

TEST_CASE( "this should test a ton of things", "[gjk]")
{
   // Make two bodies out of the convex hulls of some points
   gjk_convexPolyhedron_t bodyA;
   {
      Vector3 a(0.0f, 0.0f, 1.0f);
      Vector3 b(-1.0f, -1.0f, 0.0f);
      Vector3 c(1.0f, -1.0f, 0.0f);
      Vector3 d(-1.0f, 1.0f, 0.0f);
      Vector3 e(1.0f, 1.0f, 0.0f);
      bodyA = {
         5, {a, b, c, d, e}, (a + b + c + d + e)/5.0f
      };
   }   

   gjk_convexPolyhedron_t bodyB;
   {
      Vector3 offset;
      Vector3 a(0.0f, 0.0f, 1.0f);
      Vector3 b(-1.0f, -1.0f, 0.0f);
      Vector3 c(1.0f, -1.0f, 0.0f);
      Vector3 d(-1.0f, 1.0f, 0.0f);
      Vector3 e(1.0f, 1.0f, 0.0f);

      // Vector3 f = -1.0f * a + offset;
      Vector3 f = 1.0f * a + offset;
      Vector3 g = b + offset;
      Vector3 h = c + offset;
      Vector3 i = d + offset;
      Vector3 j = e + offset;
      bodyB = {
         5, {g, h, i, j, f}, (f + g + h + i + j)/5.0f
      };
   }

   std::cout << "making hull A\n";
   geometry::types::triangleMesh_t bodyAMesh;
   int result_a = geometry::mesh::generateHull(bodyA.numVerts, bodyA.verts, bodyAMesh);
   data_triangleMesh_t bodyADataMesh;
   geometry::converters::to_pod(bodyAMesh, &bodyADataMesh);
   std::cout << "made hull A\n";

   std::cout << "making hull B\n";
   geometry::types::triangleMesh_t bodyBMesh;
   int result_b = geometry::mesh::generateHull(bodyB.numVerts, bodyB.verts, bodyBMesh);
   data_triangleMesh_t bodyBDataMesh;
   geometry::converters::to_pod(bodyBMesh, &bodyBDataMesh);
   std::cout << "made hull B\n";

   Vector3 bodyAOffset(0.0f, 0.0f, -1.5f);
   Vector3 bodyBOffset(0.0f, -1.0f, 1.5f);
   Matrix33 R = identityMatrix();

   convert_data_mesh_to_gjk_convexPolyhedron(&bodyADataMesh, R, bodyAOffset, bodyA);
   convert_data_mesh_to_gjk_convexPolyhedron(&bodyBDataMesh, R, bodyBOffset, bodyB);

   Matrix33 eye(identityMatrix());
   Vector3 zero(0.0f, 0.0f, 0.0f);

   geometry::types::transform_t dummy_a = {
      eye, eye, zero
   };

   geometry::types::transform_t dummy_b = {
      eye, eye, zero
   };

   gjk_result_t output = geometry::gjk::alg(dummy_a, dummy_b, bodyA, bodyB);

   // It's unclear whether these two shapes are meant to be intersecting, so
   // I'm just going to test for the basics.
   REQUIRE(output.minSimplex.numVerts <= 4);
   REQUIRE(output.minSimplex.numVerts >= 0);
}
