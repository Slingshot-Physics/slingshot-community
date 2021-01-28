#include "geometry_type_converters.hpp"
#include "mesh.hpp"
#include "quickhull.hpp"
#include "random_utils.hpp"
#include "vector3.hpp"

#define CATCH_CONFIG_MAIN

#define NUM_RANDOM_CLOUD_TESTS 100

#include "catch.hpp"
#include <vector>

// void read_point_cloud_file(const char * filename, std::vector<Vector3> & point_cloud)
// {
//    FILE * fp = fopen(filename, "r");
//    if (fp == nullptr)
//    {
//       std::cout << "Couldn't open file with name: " << filename << "\n";
//       return;
//    }

//    unsigned int num_points;
//    fscanf(fp, "num points: %u\n", &num_points);

//    for (int i = 0; i < num_points; ++i)
//    {
//       data_vector3_t temp_vec3_data;
//       read_named_vector3_from_file(&temp_vec3_data, "vert", fp);
//       Vector3 temp_vec3;
//       geometry::converters::convert_data_to_Vector3(&temp_vec3_data, temp_vec3);
//       point_cloud.push_back(temp_vec3);
//    }

//    fclose(fp);
// }

TEST_CASE( "verify that simple pyramidal hulls are generated", "[quickhull]" )
{
   const int numVerts = 5;
   Vector3 points[numVerts] = {
      Vector3(-1.0f, -1.0f, 0.0f),
      Vector3(-1.0f, 1.0f, 0.0f),
      Vector3(0.5f, 0.5f, 0.0f),
      Vector3(0.0f, 0.0f, 1.0f),
      Vector3(-1.15f, -1.15f, 0.75f)
   };

   geometry::types::triangleMesh_t theHull;
   int result = geometry::mesh::generateHull(5, points, theHull);
   
   REQUIRE(theHull.numVerts == numVerts);
   REQUIRE(theHull.numTriangles == 6);
   REQUIRE(result > 0);
}

TEST_CASE( "verify that tetrahedra can be created", "[quickhull]" )
{
   Vector3 points[4] = {
      Vector3(-1.0f, -1.0f, 0.0f),
      Vector3(-1.0f, 1.0f, 0.0f),
      Vector3(0.5f, 0.5f, 0.0f),
      Vector3(0.0f, 0.0f, 1.0f)
   };

   geometry::types::triangleMesh_t theHull;
   int result = geometry::mesh::generateHull(4, points, theHull);
   
   REQUIRE(theHull.numTriangles == 4);
   REQUIRE(theHull.numVerts == 4);
   REQUIRE(result > 0);
}

// Verifies that an icosahedron can be constructed out of quickhull.
TEST_CASE( "icosahedron construction", "[quickhull]")
{
   // You are allowed to just know things.
   const unsigned int num_points = 12;
   const unsigned int num_triangles = 20;

   float phee = (1.0f + sqrtf(5.0f))/2.0f;
   float nphee = -1.0f * (1.0f + sqrtf(5.0f))/2.0f;
   Vector3 points[num_points] = {
      Vector3(0.0f, -1.0f, nphee),
      Vector3(0.0f, -1.0f, phee),
      Vector3(0.0f, 1.0f, nphee),
      Vector3(0.0f, 1.0f, phee),
      Vector3(-1.0f, nphee, 0.0f),
      Vector3(-1.0f, phee, 0.0f),
      Vector3( 1.0f, nphee, 0.0f),
      Vector3( 1.0f, phee, 0.0f),
      Vector3(-1.0f, 0.0f, nphee),
      Vector3(-1.0f, 0.0f, phee),
      Vector3( 1.0f, 0.0f, nphee),
      Vector3( 1.0f, 0.0f, phee)
   };

   geometry::types::triangleMesh_t theHull;
   int result = geometry::mesh::generateHull(
      num_points, points, theHull
   );
   REQUIRE(theHull.numTriangles == num_triangles);
   REQUIRE(theHull.numVerts == num_points);
   REQUIRE(result > 0);
}

TEST_CASE( "random clouds construction", "[quickhull]")
{
   for (int random_cloud_loop = 0; random_cloud_loop < NUM_RANDOM_CLOUD_TESTS; ++random_cloud_loop)
   {
      unsigned int num_points = rand() % MAX_VERTICES;
      std::vector<Vector3> points;

      Vector3 average_point;

      for (int i = 0; i < num_points; ++i)
      {
         points.push_back(edbdmath::random_vec3(-10.f, 10.f));

         average_point += points[i] / num_points;
      }

      geometry::types::triangleMesh_t md_hull;
      int result = geometry::mesh::generateHull(
         num_points, &(points[0]), md_hull
      );

      REQUIRE(geometry::mesh::pointInsideMesh(md_hull, average_point));
      REQUIRE( result > 0);
   }
}
