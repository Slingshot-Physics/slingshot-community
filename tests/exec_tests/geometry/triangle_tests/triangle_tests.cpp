#include "attitudeutils.hpp"
#include "random_utils.hpp"

#include "geometry_type_converters.hpp"
#include "plane.hpp"
#include "segment.hpp"
#include "triangle.hpp"

#include "test_input_serializer_util.hpp"
#include "triangle_utils.hpp"

#include <algorithm>
#include <iostream>
#include <set>

#ifndef TEST_FILENAMES
#define TEST_FILENAMES ""
#define NUM_FILENAMES 0
#endif

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#ifndef NUM_TRIANGLE_TESTS
#define NUM_TRIANGLE_TESTS 100
#endif

int dummy = edbdmath::seed_rng_ret();

Vector3 clamp_vector(float vec_min, float vec_max, const Vector3 vec)
{
   Vector3 ret;
   for (unsigned int i = 0; i < 3; ++i)
   {
      ret[i] = std::max(
         std::min(vec[i], vec_max),
         vec_min
      );
   }

   return ret;
}

int max_voronoi_region_size(const Vector3 & bary_coords)
{
   int num_verts = 0;

   for (int i = 0; i < 3; ++i)
   {
      if (bary_coords[i] > 0.f)
      {
         num_verts += 1;
      }
   }

   return num_verts;
}

// Slow algorithm for calculating the closest point on a triangle to a query
// point, q, and the Voronoi region of the triangle containing q.
Vector3 slow_closest_point_to_point(
   const geometry::types::triangle_t & triangle,
   const Vector3 & q
)
{
   Vector3 closest_point = triangle.verts[0];
   float closest_dist = (closest_point - q).magnitude();

   for (unsigned int i = 1; i < 3; ++i)
   {
      float temp_dist = (triangle.verts[i] - q).magnitude();
      if (temp_dist < closest_dist)
      {
         closest_point = triangle.verts[i];
         closest_dist = temp_dist;
      }
   }

   for (unsigned int i = 0; i < 3; ++i)
   {
      Vector3 a = triangle.verts[i];
      Vector3 b = triangle.verts[(i + 1) % 3];
      Vector3 closest_edge_point = geometry::segment::closestPointToPoint(
         a, b, q
      ).point;
      float temp_dist = (closest_edge_point - q).magnitude();
      if (temp_dist < closest_dist)
      {
         closest_point = closest_edge_point;
         closest_dist = temp_dist;
      }
   }

   Vector3 closest_plane_point = geometry::plane::closestPointToPoint(
      triangle.verts[0], triangle.verts[1], triangle.verts[2], q
   );

   Vector4 bary_coords = geometry::triangle::baryCoords(triangle, q);

   float temp_dist = (closest_plane_point - q).magnitude();
   if (
      geometry::triangle::pointInFaceVfr(
         triangle.verts[0], triangle.verts[1], triangle.verts[2], q
      )
   )
   {
      closest_point = closest_plane_point;
      closest_dist = temp_dist;
   }

   return closest_point;
}

bool test_triangle_closest_point_to_point(
   const geometry::types::testTriangleInput_t & triangle_input,
   const std::string & test_name,
   bool save_failures
)
{
   geometry::types::pointBaryCoord_t closest_bary_pt = \
      geometry::triangle::closestPointToPoint(
         triangle_input.triangle.verts[0],
         triangle_input.triangle.verts[1],
         triangle_input.triangle.verts[2],
         triangle_input.queryPoint
      );

   Vector3 slow_closest_pt = slow_closest_point_to_point(
      triangle_input.triangle, triangle_input.queryPoint
   );

   int fast_vr_size = 0;
   for (int i = 0; i < 3; ++i)
   {
      fast_vr_size += (closest_bary_pt.bary[i] > 0.f);
   }

   int max_vr_size = max_voronoi_region_size(triangle_input.queryPointBary);

   // Pitiful
   const float comparison_threshold = 1e-1f;

   const float slow_closest_dist = (slow_closest_pt - triangle_input.queryPoint).magnitude();
   const float alg_closest_dist = (closest_bary_pt.point - triangle_input.queryPoint).magnitude();

   bool requirement = (
      (closest_bary_pt.bary[0] >= 0.f) &&
      (closest_bary_pt.bary[1] >= 0.f) &&
      (closest_bary_pt.bary[2] >= 0.f) &&
      (closest_bary_pt.bary[0] <= 1.f) &&
      (closest_bary_pt.bary[1] <= 1.f) &&
      (closest_bary_pt.bary[2] <= 1.f) &&
      (
         (slow_closest_pt - closest_bary_pt.point).magnitude() / std::max(
            1e-6f,
            std::max(
               slow_closest_pt.magnitude(), closest_bary_pt.point.magnitude()
            )
         )
      ) < comparison_threshold
   );

   if (!requirement)
   {
      Vector3 normal = (
         triangle_input.triangle.verts[0] - triangle_input.triangle.verts[1]
      ).crossProduct(
         triangle_input.triangle.verts[2] - triangle_input.triangle.verts[1]
      );

      float dot = (
         (
            (triangle_input.triangle.verts[0] - triangle_input.triangle.verts[1]).unitVector()
         ).dot(
            (triangle_input.triangle.verts[2] - triangle_input.triangle.verts[1]).unitVector()
         )
      );

      std::cout << "test name: " << test_name << "\n";
      const Vector3 & test_query_point = triangle_input.queryPoint;
      std::cout << "triangle normal: " << normal << "\n";
      std::cout << "triangle dot: " << dot << "\n";
      std::cout << "bary coord: " << closest_bary_pt.bary << "\n";
      std::cout << "slow closest point: " << slow_closest_pt << "\n";
      std::cout << "alg closest point: " << closest_bary_pt.point << "\n";
      std::cout << "slow closest dist: " << slow_closest_dist << "\n";
      std::cout << "alg closest dist: " << alg_closest_dist << "\n";
      std::cout << "dist between outputs: " << (slow_closest_pt - closest_bary_pt.point).magnitude() << "\n";
      std::cout << "max number of simplex verts: " << max_vr_size << "\n";
      std::cout << "fast number of simplex verts: " << fast_vr_size << "\n";

      std::cout << "triangle verts:\n";
      for (int i = 0; i < 3; ++i)
      {
         std::cout << "\t" << triangle_input.triangle.verts[i] << "\n";
      }

      std::cout << "query point: " << triangle_input.queryPoint << "\n";

      if (save_failures)
      {
         test_utils::serialize_geometry_input<geometry::types::testTriangleInput_t, data_testTriangleInput_t>(
            triangle_input, "data_testTriangleInput_t", test_name
         );
      }
   }

   REQUIRE(
      (
         (slow_closest_pt - closest_bary_pt.point).magnitude() / std::max(
            1e-6f,
            std::max(
               slow_closest_pt.magnitude(), closest_bary_pt.point.magnitude()
            )
         )
      ) < comparison_threshold
   );

   REQUIRE( closest_bary_pt.bary[0] >= 0.f );
   REQUIRE( closest_bary_pt.bary[1] >= 0.f );
   REQUIRE( closest_bary_pt.bary[2] >= 0.f );
   REQUIRE( closest_bary_pt.bary[0] <= 1.f );
   REQUIRE( closest_bary_pt.bary[1] <= 1.f );
   REQUIRE( closest_bary_pt.bary[2] <= 1.f );

   return requirement;
}

TEST_CASE( "closest point", "[triangle]" )
{
   for (int i = 0; i < NUM_TRIANGLE_TESTS; ++i)
   {
      DYNAMIC_SECTION("closest-point-to-triangle-" << i)
      {
         std::ostringstream section_name;
         section_name << "closest-point-to-triangle-";
         section_name << i << ".json";
         geometry::types::triangle_t test_triangle = test_utils::generate_random_xy_triangle();

         Vector3 bary;
         bary[0] = edbdmath::random_float(-16.f, 16.f);
         bary[1] = edbdmath::random_float(-16.f, 16.f);
         bary[2] = 1.f - bary[0] - bary[1];

         Vector3 test_query_point = (
            bary[0] * test_triangle.verts[0] +
            bary[1] * test_triangle.verts[1] +
            bary[2] * test_triangle.verts[2]
         );

         test_query_point[2] = edbdmath::random_float(-16.f, 16.f);

         // clamp_vector(-64.f, 64.f, test_query_point);

         geometry::types::testTriangleInput_t triangle_input = {
            test_triangle, bary, test_query_point
         };

         test_triangle_closest_point_to_point(
            triangle_input, section_name.str(), true
         );
      }
   }
}

TEST_CASE( "closest point on plane", "[triangle]" )
{
   for (int i = 0; i < NUM_TRIANGLE_TESTS; ++i)
   {
      DYNAMIC_SECTION("closest-point-on-triangle-to-point-on-plane-" << i)
      {
         std::ostringstream section_name;
         section_name << "closest-point-on-triangle-to-point-on-plane-";
         section_name << i << ".json";
         geometry::types::triangle_t test_triangle = test_utils::generate_random_xy_triangle();

         Vector3 bary;
         bary[0] = edbdmath::random_float(-16.f, 16.f);
         bary[1] = edbdmath::random_float(-16.f, 16.f);
         bary[2] = 1.f - bary[0] - bary[1];

         Vector3 test_query_point = (
            bary[0] * test_triangle.verts[0] +
            bary[1] * test_triangle.verts[1] +
            bary[2] * test_triangle.verts[2]
         );

         // clamp_vector(-64.f, 64.f, test_query_point);

         geometry::types::testTriangleInput_t triangle_input = {
            test_triangle, bary, test_query_point
         };

         test_triangle_closest_point_to_point(
            triangle_input, section_name.str(), true
         );
      }
   }
}

TEST_CASE( "closest point on triangle", "[triangle]" )
{
   for (int i = 0; i < NUM_TRIANGLE_TESTS; ++i)
   {
      DYNAMIC_SECTION("closest-point-on-triangle-to-point-on-same-triangle-" << i)
      {
         std::ostringstream section_name;
         section_name << "closest-point-on-triangle-to-point-on-same-triangle-";
         section_name << i << ".json";
         geometry::types::triangle_t test_triangle = test_utils::generate_random_xy_triangle();

         Vector3 bary;
         bary[0] = edbdmath::random_float();
         bary[1] = edbdmath::random_float(0.f, bary[0]);
         bary[2] = 1.f - bary[0] - bary[1];

         Vector3 test_query_point;
         for (int i = 0; i < 3; ++i)
         {
            test_query_point += bary[i] * test_triangle.verts[i];
         }

         geometry::types::testTriangleInput_t triangle_input = {
            test_triangle, bary, test_query_point
         };

         test_triangle_closest_point_to_point(
            triangle_input, section_name.str(), true
         );
      }
   }
}

TEST_CASE( "closest point on edge", "[triangle]" )
{
   for (int i = 0; i < NUM_TRIANGLE_TESTS; ++i)
   {
      DYNAMIC_SECTION("closest-point-on-triangle-to-point-on-edge-of-same-triangle-" << i)
      {
         std::ostringstream section_name;
         section_name << "closest-point-on-triangle-to-point-on-edge-of-same-triangle-";
         section_name << i << ".json";
         geometry::types::triangle_t test_triangle = test_utils::generate_random_xy_triangle();
         int vert_a = edbdmath::random_int(0, 2);
         int vert_b = (vert_a + 1) % 3;
         float t = edbdmath::random_float();
         Vector3 test_query_point = (1.f - t) * test_triangle.verts[vert_a] + t * test_triangle.verts[vert_b];

         Vector3 bary;
         bary[vert_a] = 1.f - t;
         bary[vert_b] = t;

         geometry::types::testTriangleInput_t triangle_input = {
            test_triangle, bary, test_query_point
         };

         test_triangle_closest_point_to_point(
            triangle_input, section_name.str(), true
         );
      }
   }
}

TEST_CASE( "closest point on point", "[triangle]" )
{
   for (int i = 0; i < NUM_TRIANGLE_TESTS; ++i)
   {
      DYNAMIC_SECTION("closest-point-on-triangle-to-rando-vertex-of-same-triangle-" << i)
      {
         std::ostringstream section_name;
         section_name << "closest-point-on-triangle-to-rando-vertex-of-same-triangle-";
         section_name << i << ".json";
         geometry::types::triangle_t test_triangle = test_utils::generate_random_xy_triangle();
         unsigned int vert_idx = edbdmath::random_int(0, 2);
         Vector3 test_query_point = test_triangle.verts[vert_idx];

         Vector3 bary;
         bary[vert_idx] = 1.f;

         geometry::types::testTriangleInput_t triangle_input = {
            test_triangle, bary, test_query_point
         };

         test_triangle_closest_point_to_point(
            triangle_input, section_name.str(), true
         );
      }
   }
}

TEST_CASE( "cherry-picked test configurations", "[triangle]")
{
   std::string filenames[] = {TEST_FILENAMES};
   for (int i = 0; i < NUM_FILENAMES; ++i)
   {
      std::string section_name(filenames[i]);
      SECTION(section_name.c_str())
      {
         const std::string & filename(filenames[i]);
         std::string test_name_str(filename);
         data_testTriangleInput_t test_input_data;
         read_data_from_file(&test_input_data, filename.c_str());
         geometry::types::testTriangleInput_t test_input;
         geometry::converters::from_pod(&test_input_data, test_input);

         test_triangle_closest_point_to_point(test_input, section_name, false);
      }
   }
}
