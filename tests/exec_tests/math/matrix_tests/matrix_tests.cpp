#include "matrix.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

TEST_CASE( "template matrix ops", "[template matrix]" )
{

   Matrix<9, 9> V(
      81,
      2.0f,  10.0f, 4.0f, 0.0f, 0.0f,  0.0f, 0.0f,  0.0f,  0.0f,
      11.0f, 12.0f, 9.0f, 0.0f, 0.0f,  0.0f, 0.0f,  0.0f,  0.0f,
      13.0f, 5.0f,  6.0f, 0.0f, 0.0f,  0.0f, 0.0f,  0.0f,  0.0f,
      0.0f,  0.0f,  0.0f, 1.0f,  0.0f, 0.0f, 0.0f,  0.0f,  0.0f,
      0.0f,  0.0f,  0.0f, 0.0f,  1.0f, 0.0f, 0.0f,  0.0f,  0.0f,
      0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f,  0.0f,
      0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 7.0f,  -2.5f, 0.0f,
      0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 0.6f,  4.0f,  9.0f,
      0.0f,  0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 2.0f,  8.0f,  -3.0f
   );

   Matrix<2, 9> T(
      18,
      3.0f,	8.0f,	   9.0f,    11.0f,	6.0f,	    0.0f,   1.0f, 2.0f, -3.0f,
      10.0f,	12.0f,	7.0f,    13.0f,	14.0f,	 0.0f,   1.0f, 2.0f, -3.0f
   );

   SECTION( "slice operation" )
   {
      for (int i = 0; i < 3; ++i)
      {
         for (int j = 0; j < 3; ++j)
         {
            REQUIRE(V.getSlice<3, 3>(0, 0)(i, j) == V(i, j));
         }
      }
   }
   SECTION( "A * C * A.T" )
   {
      // Just perform the operation because... I don't want to check numbers.
      ACAtBlockDiag<3, 2, 9>(T, V);

      Matrix<2, 2> W( T * V * T.transpose());

      REQUIRE(W == ACAtBlockDiag<3, 2, 9>(T, V));
   }
   SECTION( "block matrix multiplication" )
   {
      Matrix<9, 2> U(T.transpose());

      REQUIRE(blockMatmul<3, 9, 2>(V, T.transpose()) == (V * U));
   }
}

TEST_CASE( "inverse test 3x3", "[template matrix]")
{

   Matrix<3, 3> V(
      9,
      2.0f, 10.0f, 4.0f,
      11.0f, 12.0f, 9.0f,
      13.0f, 5.0f, 6.0f
   );

   Matrix<3, 3> inv;
   invGaussElim(V, inv);

   Matrix<3, 3> near_identity = inv * V;
   Matrix<3, 3> identity;
   identity.eye();

   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         REQUIRE_THAT(
            near_identity(i, j) - identity(i, j),
            Catch::Matchers::WithinAbs(0.f, 1e-5f)
         );
      }
   }
}

TEST_CASE( "inverse test 5x5", "[template matrix]")
{
   Matrix<5, 5> T(
      25,
      3.0f,	   8.0f,	   9.0f,	   11.0f,	6.0f,	 
      10.0f,	12.0f,	7.0f,	   13.0f,	14.0f,
      15.0f,	16.0f,	4.0f,	   17.0f,	2.0f,
      18.0f,	5.0f,	   19.0f,	20.0f,	21.0f,
      22.0f,	23.0f,	24.0f,	25.0f,	26.0f
   );

   Matrix<5, 5> inv;
   invGaussElim(T, inv);

   Matrix<5, 5> near_identity = T * inv;
   Matrix<5, 5> identity;
   identity.eye();

   for (int i = 0; i < 5; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         REQUIRE_THAT(
            near_identity(i, j) - identity(i, j),
            Catch::Matchers::WithinAbs(0.f, 1e-5f)
         );
      }
   }
}

TEST_CASE( "LU decomposition", "[template matrix]")
{
   Matrix<6, 6> V = eye<6>();
   
   V(0, 0) = 2.0f;
   V(0, 1) = -3.0f;
   V(1, 0) = -3.0f;
   V(0, 3) = 2.3f;
   V(0, 5) = 4.5f;
   V(1, 2) = -4.125f;
   V(1, 3) = -3.75f;
   V(4, 5) = 9.175f;
   V(5, 4) = 2.175f;

   Matrix<6, 6> L;
   Matrix<6, 6> U;
   LUDecomposition(V, L, U);

   REQUIRE( fabs(V.determinant()) > 0.f );

   for (int i = 0; i < 6; ++i)
   {
      for (int j = 0; j < 6; ++j)
      {
         REQUIRE_THAT(
            V(i, j),
            Catch::Matchers::WithinAbs((L * U)(i, j), 1e-5f)
         );
      }
   }
}

// Just verify that it doesn't crash or segfault, I guess
TEST_CASE ( "QR decomposition", "[template matrix]")
{
   Matrix<5, 5> P(
      25,
      0.25845716f, 0.63976515f, 0.21367218f, 0.22211749f, 0.92753318f,
      0.21196188f, 0.7282288f , 0.34851991f, 0.17821241f, 0.4371361f ,
      0.70389801f, 0.43895735f, 0.77262546f, 0.99010493f, 0.37621823f,
      0.63489892f, 0.13437392f, 0.83896529f, 0.25672496f, 0.35508016f,
      0.87770761f, 0.42244043f, 0.22370777f, 0.26102438f, 0.56445663f
   );
   Matrix<5, 5> Q;
   Matrix<5, 5> R;

   QRDecomposition(P, Q, R);

   SECTION( "back-substitution" )
   {
      Matrix<5, 5> R_inv;
      backSubstitution(R, R_inv);
   }
}

// Just let it happen
TEST_CASE( "multi-fancy-ops", "[template matrix]")
{
   Matrix<5, 5> P(
      25,
      0.25845716f, 0.63976515f, 0.21367218f, 0.22211749f, 0.92753318f,
      0.21196188f, 0.7282288f , 0.34851991f, 0.17821241f, 0.4371361f ,
      0.70389801f, 0.43895735f, 0.77262546f, 0.99010493f, 0.37621823f,
      0.63489892f, 0.13437392f, 0.83896529f, 0.25672496f, 0.35508016f,
      0.87770761f, 0.42244043f, 0.22370777f, 0.26102438f, 0.56445663f
   );
   Matrix<5, 5> R;

   Matrix<5, 5> Q;

   QRDecomposition(P, Q, R);

   Matrix<5, 5> P_inv;
   inverseQR(Q, R, P_inv);

   Matrix<5, 5> S(P * P.transpose());

   QRDecomposition(S, Q, R);

   inverseQR(Q, R, P_inv);
}

// Smile and wave, boys
TEST_CASE( "slice ops", "[template matrix]" )
{
   Matrix33 eye = identityMatrix();
   Vector3 v(1.0f, 2.0f, -7.0f);
   Matrix<1, 3> v2(3, 1.0f, 2.0f, -7.0f);

   Matrix<5, 5> T(
      25,
      3.0f,	8.0f,	9.0f,	11.0f,	6.0f,	 
      10.0f,	12.0f,	7.0f,	13.0f,	14.0f,
      15.0f,	16.0f,	4.0f,	17.0f,	2.0f,
      18.0f,	5.0f,	19.0f,	20.0f,	21.0f,
      22.0f,	23.0f,	24.0f,	25.0f,	26.0f
   );

   Matrix<5, 5> Q(
      25,
      3.0f,	8.0f,	9.0f,	11.0f,	6.0f,	 
      10.0f,	12.0f,	7.0f,	13.0f,	14.0f,
      15.0f,	16.0f,	4.0f,	17.0f,	2.0f,
      18.0f,	5.0f,	19.0f,	20.0f,	21.0f,
      22.0f,	23.0f,	24.0f,	25.0f,	26.0f
   );

   Matrix<5, 5> P(
      25,
      3.0f,	8.0f,	9.0f,	11.0f,	6.0f,	 
      10.0f,	12.0f,	7.0f,	13.0f,	14.0f,
      15.0f,	16.0f,	4.0f,	17.0f,	2.0f,
      18.0f,	5.0f,	19.0f,	20.0f,	21.0f,
      22.0f,	23.0f,	24.0f,	25.0f,	26.0f
   );

   T.assignSlice(1, 1, eye);

   Q.assignSlice(1, 1, v);

   P.assignSlice(1, 1, v2);

   Matrix<3, 3> R = P.getSlice<3, 3>(1, 0);

   R = P.getSlice<3, 3>(1, 1);
}
