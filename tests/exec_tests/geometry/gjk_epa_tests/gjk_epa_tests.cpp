#include "attitudeutils.hpp"
#include "epa.hpp"
#include "gjk.hpp"
#include "logger_utils.hpp"
#include "md_hull_intersection.hpp"
#include "mesh.hpp"
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

bool test_gjk(
   const geometry::types::testGjkInput_t & test_input,
   const geometry::types::convexPolyhedron_t & polyhedronA,
   const geometry::types::convexPolyhedron_t & polyhedronB,
   const geometry::types::triangleMesh_t & md_hull,
   const std::string & test_name,
   bool serialize_on_failure,
   geometry::types::gjkResult_t & gjk_output
)
{
   gjk_output = geometry::gjk::alg(
      test_input.transformA,
      test_input.transformB,
      polyhedronA,
      polyhedronB
   );

   Vector3 zero;
   float closest_points_distance = 0.f;
   bool slow_result = geometry::mesh::pointInsideMesh(
      md_hull, zero, closest_points_distance
   );

   bool requirement = (slow_result == gjk_output.intersection);

   if (!requirement)
   {
      std::cout << "Disagreement in GJK portion of test name: " << test_name << "\n";
      std::cout << "\tdistance between closest points: " << closest_points_distance << "\n";
      if (serialize_on_failure)
      {
         std::cout << "Serializing failed input\n";
         data_testGjkInput_t test_input_data;
         geometry::converters::to_pod(test_input, &test_input_data);
         std::ostringstream output_name;
         output_name << "test-" << test_name << ".json";
         write_data_to_file(&test_input_data, "data_testGjkInput_t", output_name.str().c_str());
      }
   }

   REQUIRE( requirement );

   return requirement;
}

void test_epa(
   const geometry::types::testGjkInput_t & test_input,
   const geometry::types::convexPolyhedron_t & polyhedronA,
   const geometry::types::convexPolyhedron_t & polyhedronB,
   const geometry::types::triangleMesh_t & md_hull,
   const geometry::types::gjkResult_t & gjk_output,
   const std::string & test_name,
   bool serialize_on_failure
)
{
   Vector3 slow_normal = test_utils::epa_test(md_hull).p;

   Vector3 epa_normal = geometry::epa::alg(
      test_input.transformA,
      test_input.transformB,
      gjk_output.minSimplex,
      polyhedronA,
      polyhedronB
   ).p;

   bool results_agree = fabs(
      slow_normal.Normalize().dot(epa_normal.Normalize())
   ) >= 0.99;

   if (!results_agree)
   {
      std::cout << "Disagreement in EPA portion of test name: " << test_name << "\n";
      if (serialize_on_failure)
      {
         std::cout << "Serializing failed input\n";
         std::ostringstream output_name;
         output_name << "test-" << test_name << ".json";
         std::string filename = output_name.str();
         test_utils::serialize_geometry_input<geometry::types::testGjkInput_t, data_testGjkInput_t>(test_input, "data_testGjkInput_t", filename);
      }
   }

   REQUIRE( results_agree );
}

// The idea here is that a pair of random meshes with random transforms are
// generated, GJK is evaluated against those meshes and transforms, then the
// explicit convex hull of the Minkowski difference between the two meshes is
// generated. The test requires that if the origin is inside the explicit MD
// hull, then GJK must say there is a collision.
// This case only works for polyhedral meshes, which is why it's only executed
// against cubes.
TEST_CASE("robust cube-cube gjk and epa", "[gjk-epa]")
{
   static unsigned int num_epa_tests = 0;
   geometry::types::enumShape_t shape_types[1] = {
      geometry::types::enumShape_t::CUBE,
   };

   geometry::types::enumShape_t shapeAType;
   geometry::types::enumShape_t shapeBType;

   for (int i = 0; i < 1; ++i)
   {
      shapeAType = shape_types[i];
      for (int j = 0; j < 1; ++j)
      {
         shapeBType = shape_types[j];
         for (int k = 0; k < NUM_GJK_TESTS; ++k)
         {
            std::ostringstream test_name;
            test_name << "epa-gjk-robust-test-cube-cube-" << k;

            SECTION(test_name.str().c_str())
            {
               geometry::types::testGjkInput_t test_input;
               geometry::types::convexPolyhedron_t polyhedronA;
               geometry::types::convexPolyhedron_t polyhedronB;
               geometry::types::triangleMesh_t meshA;
               geometry::types::triangleMesh_t meshB;

               test_utils::generate_random_gjk_input(
                  shapeAType, shapeBType, test_input, polyhedronA, polyhedronB, meshA, meshB
               );

               geometry::types::triangleMesh_t md_hull;
               int result = test_utils::generateMdHull(
                  meshA, meshB, md_hull
               );

               if (result < 0)
               {
                  std::cout << "skipping\n";
                  continue;
               }

               geometry::types::gjkResult_t gjk_result;
               std::string test_name_str(test_name.str());
               bool detection_agreement = test_gjk(
                  test_input,
                  polyhedronA,
                  polyhedronB,
                  md_hull,
                  test_name_str,
                  true,
                  gjk_result
               );

               if (detection_agreement && gjk_result.intersection)
               {
                  num_epa_tests++;
                  test_epa(
                     test_input,
                     polyhedronA,
                     polyhedronB,
                     md_hull,
                     gjk_result,
                     test_name_str,
                     true
                  );
               }
            }
         }
      }
   }

   std::cout << "Total number of EPA tests " << num_epa_tests << "\n";
}
