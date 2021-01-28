#include "attitudeutils.hpp"
#include "random_utils.hpp"

#include "geometry_type_converters.hpp"
#include "mesh.hpp"
#include "plane.hpp"
#include "segment.hpp"
#include "tetrahedron.hpp"
#include "triangle.hpp"

#include "test_input_serializer_util.hpp"
#include "triangle_utils.hpp"

#include <iostream>

#ifndef TEST_FILENAMES
#define TEST_FILENAMES ""
#define NUM_FILENAMES 0
#endif

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#ifndef NUM_TETRAHEDRON_TESTS
#define NUM_TETRAHEDRON_TESTS 100
#endif

int dummy = edbdmath::seed_rng_ret();

// Generates a tetrahedron with one triangle that's on the xy plane.
geometry::types::tetrahedron_t generate_random_xy_base_tetrahedron(void)
{
   geometry::types::tetrahedron_t output;

   geometry::types::triangle_t base_triangle = test_utils::generate_random_xy_triangle();
   for (int i = 0; i < 3; ++i)
   {
      output.verts[i] = base_triangle.verts[i];
   }

   output.verts[3] = edbdmath::random_vec3(-32.f, 32.f, -32.f, 32.f, 1e-3f, 32.f);
   float top_vert_z_sign = edbdmath::random_float() * 2 - 1.f;
   output.verts[3][2] *= top_vert_z_sign;

   return output;
}

int max_voronoi_region_size(const Vector4 & bary_coords)
{
   int num_verts = 0;

   for (int i = 0; i < 4; ++i)
   {
      if (bary_coords[i] > 0.f)
      {
         num_verts += 1;
      }
   }

   return num_verts;
}

Vector3 slow_closest_point_to_point(
   const geometry::types::tetrahedron_t & tetrahedron, const Vector3 & q
)
{
   Vector3 closest_point = tetrahedron.verts[0];
   float closest_dist = (closest_point - q).magnitude();

   for (int i = 1; i < 4; ++i)
   {
      float temp_dist = (tetrahedron.verts[i] - q).magnitude();
      if (temp_dist < closest_dist)
      {
         closest_point = tetrahedron.verts[i];
         closest_dist = temp_dist;
      }
   }

   for (int i = 0; i < 4; ++i)
   {
      Vector3 a = tetrahedron.verts[i];
      for (int j = i; j < 4; ++j)
      {
         Vector3 b = tetrahedron.verts[j];
         if (i == j)
         {
            continue;
         }

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
   }

   for (int i = 0; i < 4; ++i)
   {
      const Vector3 & vert_a = tetrahedron.verts[geometry::tetrahedron::faceCombos[i].vertAId];
      const Vector3 & vert_b = tetrahedron.verts[geometry::tetrahedron::faceCombos[i].vertBId];
      const Vector3 & vert_c = tetrahedron.verts[geometry::tetrahedron::faceCombos[i].vertCId];

      Vector3 closest_plane_point = geometry::plane::closestPointToPoint(
         vert_a, vert_b, vert_c, q
      );

      geometry::types::triangle_t triangle = {
         {vert_a, vert_b, vert_c}
      };

      Vector4 bary_coords = geometry::triangle::baryCoords(triangle, q);

      float temp_dist = (closest_plane_point - q).magnitude();

      if (
         (temp_dist < closest_dist) &&
         (bary_coords[0] >= 0.f) &&
         (bary_coords[1] >= 0.f) &&
         (bary_coords[2] >= 0.f) &&
         (bary_coords[0] <= 1.f) &&
         (bary_coords[1] <= 1.f) &&
         (bary_coords[2] <= 1.f)
      )
      {
         closest_point = closest_plane_point;
         closest_dist = temp_dist;
      }
   }

   if (
      geometry::tetrahedron::pointInside(
         tetrahedron.verts[0],
         tetrahedron.verts[1],
         tetrahedron.verts[2],
         tetrahedron.verts[3],
         q
      )
   )
   {
      closest_point = q;
   }

   return closest_point;
}

void test_tetrahedron_closest_point_to_point(
   const geometry::types::testTetrahedronInput_t & tetrahedron_input,
   const std::string & test_name,
   bool save_on_fail
)
{
   geometry::types::pointBaryCoord_t closest_bary_pt = geometry::tetrahedron::closestPointToPoint(
      tetrahedron_input.tetrahedron.verts[0],
      tetrahedron_input.tetrahedron.verts[1],
      tetrahedron_input.tetrahedron.verts[2],
      tetrahedron_input.tetrahedron.verts[3],
      tetrahedron_input.queryPoint
   );

   Vector3 slow_closest_pt = slow_closest_point_to_point(
      tetrahedron_input.tetrahedron, tetrahedron_input.queryPoint
   );

   int fast_vr_size = 0;
   for (int i = 0; i < 3; ++i)
   {
      fast_vr_size += (closest_bary_pt.bary[i] > 0.f);
   }

   int max_vr_size = max_voronoi_region_size(tetrahedron_input.queryPointBary);

   const float comparison_threshold = 1e-5f;

   const Vector3 & test_query_point = tetrahedron_input.queryPoint;
   float slow_dist = (slow_closest_pt - test_query_point).magnitude();
   float fast_dist = (closest_bary_pt.point - test_query_point).magnitude();
   float rel_dist_diff = fabs(slow_dist - fast_dist) / fmaxf(fmaxf(slow_dist, fast_dist), 1e-7f);
   float rel_dist = (slow_closest_pt - closest_bary_pt.point).magnitude();
   bool requirement = (
      (!closest_bary_pt.bary.hasNan()) && 
      (!closest_bary_pt.bary.hasInf()) &&
      (
         (rel_dist_diff <= 0.1f) ||
         (fast_dist <= slow_dist) ||
         (rel_dist <= comparison_threshold) ||
         ((slow_dist <= 1e-5f) && (fast_dist <= 1e-3f)) // the query point is *on* the tetrahedron
      )
   );

   if (!requirement)
   {
      std::cout << "fast bary coord: " << closest_bary_pt.bary << "\n";
      std::cout << "query point bary coord: " << tetrahedron_input.queryPointBary << "\n";
      std::cout << "slow closest: " << slow_closest_pt << "\n";
      std::cout << "alg closest: " << closest_bary_pt.point << "\n";
      std::cout << "alg bary coord: " << closest_bary_pt.bary << "\n";
      std::cout << "dist between outputs: " << (slow_closest_pt - closest_bary_pt.point).magnitude() << "\n";
      std::cout << "slow closest dist: " << slow_dist << "\n";
      std::cout << "alg closest dist: " << fast_dist << "\n";
      std::cout << "relative difference between distances: " << rel_dist_diff << "\n";
      std::cout << "max number of simplex verts: " << max_vr_size << "\n";
      std::cout << "fast number of simplex verts: " << fast_vr_size << "\n";

      if (save_on_fail)
      {
         test_utils::serialize_geometry_input<geometry::types::testTetrahedronInput_t, data_testTetrahedronInput_t>(
            tetrahedron_input, "data_testTetrahedronInput_t", test_name
         );
      }
   }

   REQUIRE(
      (
         (rel_dist_diff <= 0.1f) ||
         (fast_dist <= slow_dist) ||
         (rel_dist <= comparison_threshold) ||
         ((slow_dist <= 1e-5f) && (fast_dist <= 1e-3f)) // the query point is *on* the tetrahedron
      )
   );
   REQUIRE( !closest_bary_pt.bary.hasNan() );
   REQUIRE( !closest_bary_pt.bary.hasInf() );
}

TEST_CASE( "closest point to random point", "[tetrahedron]" )
{
   for (int i = 0; i < NUM_TETRAHEDRON_TESTS; ++i)
   {
      DYNAMIC_SECTION("tetrahedron-closest-point-to-random-point-" << i)
      {
         std::ostringstream filename;
         filename << "tetrahedron-closest-point-to-random-point-" << i << ".json";
         geometry::types::testTetrahedronInput_t test_input;
         test_input.tetrahedron = generate_random_xy_base_tetrahedron();

         test_input.queryPointBary = edbdmath::random_vec4(-5.f, 5.f);
         test_input.queryPointBary[3] = (
            1.f - test_input.queryPointBary[0] - test_input.queryPointBary[1] - test_input.queryPointBary[2]
         );

         test_input.queryPoint = (
            test_input.queryPointBary[0] * test_input.tetrahedron.verts[0] +
            test_input.queryPointBary[1] * test_input.tetrahedron.verts[1] +
            test_input.queryPointBary[2] * test_input.tetrahedron.verts[2] +
            test_input.queryPointBary[3] * test_input.tetrahedron.verts[3]
         );

         test_tetrahedron_closest_point_to_point(test_input, filename.str(), true);
      }
   }
}

TEST_CASE( "closest point to point on edge", "[tetrahedron]" )
{
   for (int i = 0; i < NUM_TETRAHEDRON_TESTS; ++i)
   {
      DYNAMIC_SECTION("tetrahedron-closest-point-to-edge-point-" << i)
      {
         std::ostringstream filename;
         filename << "tetrahedron-closest-point-to-edge-point-" << i << ".json";

         int vert_idx_a = edbdmath::random_int(0, 3);
         int vert_idx_b = (
            vert_idx_a + edbdmath::random_int(0, 2)
         ) % 4;
         float t = edbdmath::random_float();

         geometry::types::tetrahedron_t test_tetra = generate_random_xy_base_tetrahedron();

         Vector3 query_point = (
            (1.f - t) * test_tetra.verts[vert_idx_a] +
            t * test_tetra.verts[vert_idx_b]
         );

         Vector4 edge_bary_coord;
         edge_bary_coord[vert_idx_a] = (1.f - t);

         edge_bary_coord[vert_idx_b] = t;

         geometry::types::testTetrahedronInput_t test_input = {
            test_tetra, edge_bary_coord, query_point
         };

         test_tetrahedron_closest_point_to_point(test_input, filename.str(), true);
      }
   }
}

TEST_CASE( "closest point to point on face", "[tetrahedron]" )
{
   for (int i = 0; i < NUM_TETRAHEDRON_TESTS; ++i)
   {
      DYNAMIC_SECTION("tetrahedron-closest-point-to-face-point-" << i)
      {
         std::ostringstream filename;
         filename << "tetrahedron-closest-point-to-face-point-" << i << ".json";

         geometry::types::tetrahedron_t test_tetra = generate_random_xy_base_tetrahedron();

         int face_idx = edbdmath::random_int(0, 3);

         int vert_ids[4] = {
            geometry::tetrahedron::faceCombos[face_idx].vertAId,
            geometry::tetrahedron::faceCombos[face_idx].vertBId,
            geometry::tetrahedron::faceCombos[face_idx].vertCId,
            geometry::tetrahedron::faceCombos[face_idx].vertDId
         };

         Vector4 face_bary_coord;
         face_bary_coord[vert_ids[0]] = edbdmath::random_float(1e-5f, 1.f);
         face_bary_coord[vert_ids[1]] = edbdmath::random_float(0.f, 1.f - face_bary_coord[vert_ids[0]]);
         face_bary_coord[vert_ids[2]] = 1.f - face_bary_coord[vert_ids[0]] - face_bary_coord[vert_ids[1]];

         Vector3 query_point = (
            face_bary_coord[0] * test_tetra.verts[0] +
            face_bary_coord[1] * test_tetra.verts[1] +
            face_bary_coord[2] * test_tetra.verts[2] +
            face_bary_coord[3] * test_tetra.verts[3]
         );

         geometry::types::testTetrahedronInput_t test_input = {
            test_tetra, face_bary_coord, query_point
         };

         test_tetrahedron_closest_point_to_point(test_input, filename.str(), true);
      }
   }
}

TEST_CASE( "closest point to point on tetrahdron", "[tetrahedron]" )
{
   for (int i = 0; i < NUM_TETRAHEDRON_TESTS; ++i)
   {
      DYNAMIC_SECTION("tetrahedron-closest-point-to-tetrahedron-point-" << i)
      {
         std::ostringstream filename;
         filename << "tetrahedron-closest-point-to-tetrahedron-point-" << i << ".json";

         geometry::types::tetrahedron_t test_tetra = generate_random_xy_base_tetrahedron();

         Vector4 tetra_bary_coord;
         tetra_bary_coord[0] = edbdmath::random_float();
         tetra_bary_coord[1] = edbdmath::random_float(0.f, 1.f - tetra_bary_coord[0]);
         tetra_bary_coord[2] = edbdmath::random_float(0.f, 1.f - tetra_bary_coord[0] - tetra_bary_coord[1]);
         tetra_bary_coord[3] = edbdmath::random_float(0.f, 1.f - tetra_bary_coord[0] - tetra_bary_coord[1] - tetra_bary_coord[2]);

         Vector3 query_point = (
            tetra_bary_coord[0] * test_tetra.verts[0] +
            tetra_bary_coord[1] * test_tetra.verts[1] +
            tetra_bary_coord[2] * test_tetra.verts[2] +
            tetra_bary_coord[3] * test_tetra.verts[3]
         );

         geometry::types::testTetrahedronInput_t test_input = {
            test_tetra, tetra_bary_coord, query_point
         };

         test_tetrahedron_closest_point_to_point(test_input, filename.str(), true);
      }
   }
}

TEST_CASE( "cherry-picked tests", "[tetrahedron]" )
{
   std::string filenames[] = {TEST_FILENAMES};
   for (int i = 0; i < NUM_FILENAMES; ++i)
   {
      std::string section_name(filenames[i]);
      SECTION(section_name.c_str())
      {
         const std::string & filename(filenames[i]);
         std::string test_name_str(filename);
         data_testTetrahedronInput_t test_input_data;
         read_data_from_file(&test_input_data, filename.c_str());
         geometry::types::testTetrahedronInput_t test_input;
         geometry::converters::from_pod(&test_input_data, test_input);

         test_tetrahedron_closest_point_to_point(test_input, section_name, false);
      }
   }
}
