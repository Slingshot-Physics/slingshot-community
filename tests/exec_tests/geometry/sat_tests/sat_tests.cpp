#include "attitudeutils.hpp"
#include "geometry_types.hpp"
#include "matrix33.hpp"
#include "vector3.hpp"
#include <math.h>

#include "sat.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

// Verify that a positive collision produces non-insane results.
TEST_CASE("positive sphere-sphere intersection")
{
   geometry::types::shapeSphere_t sphere_a{2.f};
   geometry::types::shapeSphere_t sphere_b{3.f};

   Vector3 pos_a = Vector3(2.f, 0.f, 0.f);
   Vector3 pos_b = Vector3(0.f, -1.f, 0.f);

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), pos_a
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), pos_b
   };

   geometry::types::satResult_t output = geometry::collisions::sphereSphere(
      trans_A_to_W, trans_B_to_W, sphere_a, sphere_b
   );

   REQUIRE(output.collision);
   REQUIRE(output.contactNormal.dot(pos_b - pos_a) > 0.f);
}

// Verify that a negative collision produces non-insane results.
TEST_CASE("negative sphere-sphere intersection")
{
   geometry::types::shapeSphere_t sphere_a{2.f};
   geometry::types::shapeSphere_t sphere_b{3.f};

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {2.f, 3.f, 0.f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {10.f, -1.f, 0.f}
   };

   geometry::types::satResult_t output = geometry::collisions::sphereSphere(
      trans_A_to_W, trans_B_to_W, sphere_a, sphere_b
   );

   REQUIRE(!output.collision);
   REQUIRE(output.contactNormal.magnitude() > 1e-7f);
   REQUIRE(output.deepestPointsA[0].magnitude() > 1e-7f);
   REQUIRE(output.deepestPointsB[0].magnitude() > 1e-7f);
}

// Verify that a positive capsule-capsule collision produces non-insane results.
TEST_CASE("positive capsule-capsule intersection")
{
   geometry::types::shapeCapsule_t cap_a;
   cap_a.radius = 2.f;
   cap_a.height = 3.f;
   geometry::types::shapeCapsule_t cap_b;
   cap_b.radius = 1.f;
   cap_b.height = 4.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {1.f, 0.f, 0.f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {-1.f, 0.f, 0.f}
   };

   geometry::types::satResult_t output = geometry::collisions::capsuleCapsule(
      trans_A_to_W, trans_B_to_W, cap_a, cap_b
   );

   REQUIRE( output.collision );
   REQUIRE( output.numDeepestPointPairs == 2 );
}

// Verify that a negative capsule-capsule collision produces non-insane results.
TEST_CASE("negative capsule-capsule intersection")
{
   geometry::types::shapeCapsule_t cap_a;
   cap_a.radius = 2.f;
   cap_a.height = 3.f;
   geometry::types::shapeCapsule_t cap_b;
   cap_b.radius = 1.f;
   cap_b.height = 4.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {4.f, 0.f, 0.f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {-6.f, 0.f, 0.f}
   };

   geometry::types::satResult_t output = geometry::collisions::capsuleCapsule(
      trans_A_to_W, trans_B_to_W, cap_a, cap_b
   );

   REQUIRE( !output.collision );
   REQUIRE( output.numDeepestPointPairs == 2 );
}

// Verify that a positive capsule-sphere collision produces non-insane results.
TEST_CASE( "positive capsule-sphere intersection" )
{
   geometry::types::shapeCapsule_t cap;
   cap.radius = 2.f;
   cap.height = 3.f;

   geometry::types::shapeSphere_t sphere{2.f};

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {1.5f, 0.f, 0.4f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {-1.5f, 0.f, -0.4f}
   };

   geometry::types::satResult_t output = geometry::collisions::sphereCapsule(
      trans_A_to_W, trans_B_to_W, sphere, cap
   );

   REQUIRE( output.collision );
}

// Verify that a negative capsule-sphere collision produces non-insane results.
TEST_CASE( "negative capsule-sphere intersection" )
{
   geometry::types::shapeCapsule_t cap;
   cap.radius = 2.f;
   cap.height = 3.f;

   geometry::types::shapeSphere_t sphere{2.f};

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {2.5f, 0.f, 0.4f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {-2.5f, 0.f, -0.4f}
   };

   geometry::types::satResult_t output = geometry::collisions::sphereCapsule(
      trans_A_to_W, trans_B_to_W, sphere, cap
   );

   REQUIRE( !output.collision );
}

// Places two cubes in a penetrating configuration in the z-hat direction.
// Cube A is on top of cube B.
TEST_CASE( "positive cube-cube intersection" )
{
   geometry::types::shapeCube_t cube_a;
   cube_a.height = 1.f;
   cube_a.length = 1.f;
   cube_a.width = 1.f;

   geometry::types::shapeCube_t cube_b;
   cube_b.height = 1.f;
   cube_b.length = 1.f;
   cube_b.width = 1.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {0.f, 0.f, 0.25f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {0.f, 0.f, -0.25f}
   };

   geometry::types::satResult_t output = geometry::collisions::cubeCube(
      trans_A_to_W, trans_B_to_W, cube_a, cube_b
   );

   Vector3 z_hat(0.f, 0.f, 1.f);
   REQUIRE( output.collision );
   REQUIRE( std::fabs(output.contactNormal.unitVector().dot(z_hat)) >= 1.f - 1e-6f );
}

// Places two cubes in a separated configuration in the z-hat direction.
// Cube A is far above cube B.
TEST_CASE( "negative cube-cube intersection" )
{
   geometry::types::shapeCube_t cube_a;
   cube_a.height = 1.f;
   cube_a.length = 1.f;
   cube_a.width = 1.f;

   geometry::types::shapeCube_t cube_b;
   cube_b.height = 1.f;
   cube_b.length = 1.f;
   cube_b.width = 1.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {0.f, 0.f, 2.25f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {0.f, 0.f, -2.25f}
   };

   geometry::types::satResult_t output = geometry::collisions::cubeCube(
      trans_A_to_W, trans_B_to_W, cube_a, cube_b
   );

   Vector3 z_hat(0.f, 0.f, 1.f);
   REQUIRE( !output.collision );

}

// Places two cubes in a penetrating configuration in the z-hat direction.
// Cube A is on top of cube B. Cube A is pretty big (10 x 10 x 10).
TEST_CASE( "positive big cube-cube intersection" )
{
   geometry::types::shapeCube_t cube_a;
   cube_a.height = 10.f;
   cube_a.length = 10.f;
   cube_a.width = 10.f;

   geometry::types::shapeCube_t cube_b;
   cube_b.height = 1.f;
   cube_b.length = 1.f;
   cube_b.width = 1.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {0.f, 0.f, 4.f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {0.f, 0.f, -0.5f}
   };

   geometry::types::satResult_t output = geometry::collisions::cubeCube(
      trans_A_to_W, trans_B_to_W, cube_a, cube_b
   );

   Vector3 z_hat(0.f, 0.f, 1.f);
   REQUIRE( output.collision );
   REQUIRE( std::fabs(output.contactNormal.unitVector().dot(z_hat)) >= 1.f - 1e-6f );
}

// Places two cubes in a penetrating configuration in the xy-plane direction.
// Cube A is diagonally offset from cube B. The only requirement is that the
// contact normal has no component in the z-axis.
TEST_CASE( "positive corner cube-cube intersection" )
{
   geometry::types::shapeCube_t cube_a;
   cube_a.height = 1.f;
   cube_a.length = 1.f;
   cube_a.width = 1.f;

   geometry::types::shapeCube_t cube_b;
   cube_b.height = 1.f;
   cube_b.length = 1.f;
   cube_b.width = 1.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {-0.5f, -0.5f, 0.f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {0.45f, 0.45f, 0.f}
   };

   geometry::types::satResult_t output = geometry::collisions::cubeCube(
      trans_A_to_W, trans_B_to_W, cube_a, cube_b
   );

   Vector3 z_hat(0.f, 0.f, 1.f);
   REQUIRE( output.collision );
   REQUIRE( std::fabs(output.contactNormal[2]) < 1e-7f );
}

// Places two cubes in a penetrating configuration in the z-hat direction.
// Cube A is on top of cube B.
TEST_CASE( "positive face cube-cube intersection" )
{
   geometry::types::shapeCube_t cube_a;
   cube_a.height = 1.f;
   cube_a.length = 1.f;
   cube_a.width = 1.f;

   geometry::types::shapeCube_t cube_b;
   cube_b.height = 1.f;
   cube_b.length = 1.f;
   cube_b.width = 1.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {-0.5f, -0.5f, 0.f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {0.5f, 0.5f, 0.5f}
   };

   geometry::types::satResult_t output = geometry::collisions::cubeCube(
      trans_A_to_W, trans_B_to_W, cube_a, cube_b
   );

   Vector3 z_hat(0.f, 0.f, 1.f);
   REQUIRE( output.collision );
   std::cout << "collision normal: " << output.contactNormal << "\n";
}

TEST_CASE( "positive cube-sphere intersection" )
{
   geometry::types::shapeCube_t cube_a;
   cube_a.height = 2.f;
   cube_a.length = 2.f;
   cube_a.width = 2.f;

   geometry::types::shapeSphere_t sphere_b;
   sphere_b.radius = 1.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {-0.5f, -0.5f, 0.f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {0.5f, 0.5f, 0.5f}
   };

   geometry::types::satResult_t output = geometry::collisions::cubeSphere(
      trans_A_to_W, trans_B_to_W, cube_a, sphere_b
   );

   Vector3 z_hat(0.f, 0.f, 1.f);
   REQUIRE( output.collision );
   std::cout << "collision normal: " << output.contactNormal << "\n";
}

TEST_CASE( "negative cube-sphere intersection" )
{
   geometry::types::shapeCube_t cube_a;
   cube_a.height = 2.f;
   cube_a.length = 2.f;
   cube_a.width = 2.f;

   geometry::types::shapeSphere_t sphere_b;
   sphere_b.radius = 1.f;

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {-0.5f, -0.5f, 10.f}
   };

   geometry::types::isometricTransform_t trans_B_to_W = {
      identityMatrix(), {0.5f, 0.5f, 0.5f}
   };

   geometry::types::satResult_t output = geometry::collisions::cubeSphere(
      trans_A_to_W, trans_B_to_W, cube_a, sphere_b
   );

   REQUIRE( !output.collision );
}
