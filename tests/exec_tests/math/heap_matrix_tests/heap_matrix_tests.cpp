#include "heap_matrix.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

// Verify that multiplying two identity matrices yields an identity matrix.
TEST_CASE( "matmul test", "[heapmatrix]")
{
   HeapMatrix m(5, 5);
   HeapMatrix n(5, 5);
   n.eye();
   m.eye();

   HeapMatrix p(5, 5);
   p = m * n;

   HeapMatrix q(5, 5);
   q.eye();

   for (int i = 0; i < 5; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         REQUIRE (p(i, j) == q(i, j));
      }
   }
}

TEST_CASE( "multi-ops", "[heapmatrix]" )
{
   HeapMatrix m(5, 5);
   HeapMatrix n(5, 5);
   n.eye();
   m.eye();

   m *= 2;
   n += 0.5f;

   std::cout << m << "\n";

   std::cout << n << "\n";

   HeapMatrix p(5, 5);
   std::cout << "p: " << p << "\n";
   p = m * n;

   for (int i = 0; i < 5; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         REQUIRE(p(i, j) != 0.f);
      }
   }
}

TEST_CASE( "matrix-vector mult", "[heapmatrix]" )
{
   HeapMatrix m(5, 5);
   HeapMatrix n(5, 1);
   m.eye();

   for (unsigned int i = 0; i < n.num_rows; ++i)
   {
      n(i, 0) = i / 3.0f;
   }

   m += 0.2f;
   n += 0.5f;

   for (int i = 0; i < 5; ++i)
   {
      REQUIRE(n(i, 0) >= 0.5f);
      for (int j = 0; j < 5; ++j)
      {
         REQUIRE(m(i, j) >= 0.2f);
      }
   }

   HeapMatrix p(5, 1);
   p = m * n;

   for (int i = 0; i < 5; ++i)
   {
      REQUIRE(p(i, 0) > 0.f);
   }
}

TEST_CASE( "vector-matrix mult", "[heapmatrix]" )
{
   HeapMatrix m(1, 5);
   HeapMatrix n(5, 5);
   n.eye();

   for (unsigned int i = 0; i < n.num_rows; ++i)
   {
      m(0, i) = i / 3.0f;
   }

   m += 0.2f;
   n += 0.5f;

   for (int i = 0; i < 5; ++i)
   {
      REQUIRE(m(0, i) >= 0.2f);
      for (int j = 0; j < 5; ++j)
      {
         REQUIRE(n(i, j) >= 0.5f);
      }
   }

   HeapMatrix p(1, 5);
   p = m * n;

   for (int i = 0; i < 5; ++i)
   {
      REQUIRE(p(0, i) > 0.f);
   }
}

// Verifies that A * B = (A.T * B.T).T for large matrices
TEST_CASE( "big matrix ops", "[heapmatrix]" )
{
   HeapMatrix m(40, 100);
   HeapMatrix n(100, 20);

   for (unsigned int i = 0; i < m.num_rows; ++i)
   {
      for (unsigned int j = 0; j < m.num_cols; ++j)
      {
         m(i, j) = ((float )std::rand()) / RAND_MAX;
      }
   }

   for (unsigned int i = 0; i < n.num_rows; ++i)
   {
      for (unsigned int j = 0; j < n.num_cols; ++j)
      {
         n(i, j) = ((float )std::rand()) / RAND_MAX;
      }
   }

   HeapMatrix p(40, 20);
   p = m * n;

   HeapMatrix q(20, 40);
   q = n.transpose() * m.transpose();

   REQUIRE(p == q.transpose());
}

TEST_CASE( "inner product with row", "[heapmatrix]" )
{
   HeapMatrix m(40, 100);
   HeapMatrix n(100, 1);

   for (unsigned int i = 0; i < 100; ++i)
   {
      n(i, 0) = (float )i;
      for (unsigned int j = 0; j < 40; ++j)
      {
         m(j, i) = 1.f * j;
      }
   }

   HeapMatrix q(m * n);

   float p = m.innerProductRowVec(20, n);

   REQUIRE( fabs(p - q(20, 0)) < 1e-5f );
}

// Designed to duplicate a bug where multiplying small matrices with SSE
// resulted in zeroed out matrices.
TEST_CASE( "inner product with row sse", "[heapmatrix]" )
{
   HeapMatrix m(6, 1);
   HeapMatrix n(1, 1);

   n(0, 0) = -2.f;
   m(1, 0) = 1.f;

   HeapMatrix q(m * n);

   float p[6];

   for (int i = 0; i < 6; ++i)
   {
      p[i] = m.innerProductRowVecSse(i, n);
   }

   for (int i = 0; i < 6; ++i)
   {
      REQUIRE( std::fabs(p[i] - q(i, 0)) < 1e-7f );
   }
}
