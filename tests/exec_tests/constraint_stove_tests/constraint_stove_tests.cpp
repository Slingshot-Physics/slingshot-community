// The base use case is:
//    1. Generate a constraint graph
//       a. generate a subgraph
//       b. pull any cached values out of the stove for that subgraph
//       c. calculate lambda_dt
//       d. update cached values with new lambda_dt and current subgraph
//       e. go to step 1.a. until the constraint graph is exhausted
//    2. Increment the stove
//    3. Step simulation, go to step 1

#include "constraint_stove.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "dynamic_array.hpp"
#include "slingshot_types.hpp"
#include "graph.hpp"
#include "heap_matrix.hpp"
#include "random_utils.hpp"
#include "subgraph.hpp"

#include <iostream>

typedef graph::types::labeled_edge_t<trecs::uid_t, oy::types::edgeId_t> stoveEdge_t;

TEST_CASE( "well i do declare", "[stove]" )
{
   oy::ConstraintStove stove(5);

   REQUIRE(true);
}

// Adds the equivalent of one timesteps' worth of data from a constraint graph
// with completely unique entries.
TEST_CASE( "add one set of unique values", "[stove]" )
{
   unsigned int num_constraints = 5;
   DynamicArray<stoveEdge_t> temp_graph(32);
   HeapMatrix temp_lagrange(num_constraints, 1);
   for (unsigned int i = 0; i < num_constraints; ++i)
   {
      stoveEdge_t temp_edge;
      temp_edge.nodeIdA = i;
      temp_edge.nodeIdB = 2 * i + 1;
      temp_edge.edgeId.constraintType = static_cast<oy::types::enumConstraint_t>(i % 7 + 1);
      temp_edge.edgeId.entity = 100;
      temp_edge.edgeId.offset = i + 100;

      std::cout << "type: " << static_cast<int>(temp_edge.edgeId.constraintType) << ", a: " << temp_edge.nodeIdA << ", b: " << temp_edge.nodeIdB << "\n";

      temp_graph.push_back(temp_edge);
      temp_lagrange(i, 0) = static_cast<float>(i);
   }

   oy::ConstraintStove stove(5);

   graph::types::subgraph_bookmark_t bookmark = {0, static_cast<unsigned int>(temp_graph.size())};

   graph::Subgraph<trecs::uid_t, oy::types::edgeId_t> subgraph(temp_graph, bookmark);

   stove.push(temp_lagrange, subgraph);

   HeapMatrix pull_lagrange(num_constraints, 1);
   stove.pull(pull_lagrange, subgraph, 0.01f);

   for (unsigned int i = 0; i < num_constraints; ++i)
   {
      REQUIRE(pull_lagrange(i, 0) == temp_lagrange(i, 0));
   }
}

TEST_CASE( "equality constraints are not added to the stove", "[stove]")
{
   unsigned int num_constraints = 6;
   DynamicArray<stoveEdge_t> temp_graph(32);
   HeapMatrix temp_lagrange(num_constraints, 1);
   for (unsigned int i = 0; i < num_constraints; ++i)
   {
      stoveEdge_t temp_edge;
      temp_edge.nodeIdA = i;
      temp_edge.nodeIdB = 2 * i + 1;
      temp_edge.edgeId.constraintType = static_cast<oy::types::enumConstraint_t>(i % 7 + 1);
      temp_edge.edgeId.entity = i;
      // Offset of -1 indicates that the constraint referenced by the edge
      // is an equality constraint - equality constraints aren't warm-started
      // because they can cause the sim to become unstable.
      temp_edge.edgeId.offset = -1;

      temp_graph.push_back(temp_edge);
      temp_lagrange(i, 0) = static_cast<float>(i);
   }

   oy::ConstraintStove stove(5);

   graph::types::subgraph_bookmark_t bookmark = {
      0, static_cast<unsigned int>(temp_graph.size())
   };

   graph::Subgraph<trecs::uid_t, oy::types::edgeId_t> subgraph(temp_graph, bookmark);

   stove.push(temp_lagrange, subgraph);

   // There should be zero entries in the stove
   REQUIRE(stove.size() == 0);
}

// Generates a series of unique entries in the constraint graph, but overwrites
// the last entry with the first entry in the graph.
TEST_CASE( "add a series of values with non-unique entries", "[stove]" )
{
   unsigned int num_constraints = 6;
   DynamicArray<stoveEdge_t> temp_graph(32);
   HeapMatrix temp_lagrange(num_constraints, 1);
   for (unsigned int i = 0; i < num_constraints; ++i)
   {
      stoveEdge_t temp_edge;
      temp_edge.nodeIdA = i;
      temp_edge.nodeIdB = 2 * i + 1;
      temp_edge.edgeId.constraintType = static_cast<oy::types::enumConstraint_t>(i % 7 + 1);
      temp_edge.edgeId.entity = i;
      temp_edge.edgeId.offset = 0;

      std::cout << "type: " << static_cast<int>(temp_edge.edgeId.constraintType) << ", a: " << temp_edge.nodeIdA << ", b: " << temp_edge.nodeIdB << "\n";

      temp_graph.push_back(temp_edge);
      temp_lagrange(i, 0) = static_cast<float>(i);
   }

   // Overwrite the last entry of the constraint graph with the first entry.
   // This simulates multiple contacts between two bodies.
   temp_graph.back().nodeIdA = temp_graph.front().nodeIdA;
   temp_graph.back().nodeIdB = temp_graph.front().nodeIdB;
   temp_graph.back().edgeId.constraintType = temp_graph.front().edgeId.constraintType;

   oy::ConstraintStove stove(5);

   graph::types::subgraph_bookmark_t bookmark = {
      0, static_cast<unsigned int>(temp_graph.size())
   };

   graph::Subgraph<trecs::uid_t, oy::types::edgeId_t> subgraph(temp_graph, bookmark);

   stove.push(temp_lagrange, subgraph);

   // There should only be five unique entries in the stove since two of the
   // graph edges have the same constraint type and body ID's.
   REQUIRE(stove.size() == 5);

   HeapMatrix pull_lagrange(num_constraints, 1);
   stove.pull(pull_lagrange, subgraph, 0.01f);

   for (unsigned int i = 0; i < num_constraints; ++i)
   {
      if (i != 0 && i != num_constraints - 1)
      {
         REQUIRE(pull_lagrange(i, 0) == temp_lagrange(i, 0));
      }
      else
      {
         REQUIRE(pull_lagrange(i, 0) == (temp_lagrange(0, 0) + temp_lagrange(5, 0))/2.f);
      }
   }
}

// Verifies that a single value lives in the stove for the lifetime defined
// in the stove's instantiation.
TEST_CASE( "pruning works", "[stove]" )
{
   unsigned int num_constraints = 1;
   DynamicArray<stoveEdge_t> temp_graph(32);
   HeapMatrix temp_lagrange(num_constraints, 1);
   for (unsigned int i = 0; i < num_constraints; ++i)
   {
      stoveEdge_t temp_edge;
      temp_edge.nodeIdA = i;
      temp_edge.nodeIdB = 2 * i + 1;
      temp_edge.edgeId.constraintType = static_cast<oy::types::enumConstraint_t>(i % 7 + 1);
      temp_edge.edgeId.entity = 8;
      temp_edge.edgeId.offset = i + 9;

      std::cout << "type: " << static_cast<int>(temp_edge.edgeId.constraintType) << ", a: " << temp_edge.nodeIdA << ", b: " << temp_edge.nodeIdB << "\n";

      temp_graph.push_back(temp_edge);
      temp_lagrange(i, 0) = static_cast<float>(i);
   }

   // Entries in the stove have a lifetime of five increments.
   oy::ConstraintStove stove(5);

   graph::types::subgraph_bookmark_t bookmark = {0, static_cast<unsigned int>(temp_graph.size())};

   graph::Subgraph<trecs::uid_t, oy::types::edgeId_t> subgraph(temp_graph, bookmark);

   // Push one lagrange entry onto the stove and increment.
   stove.push(temp_lagrange, subgraph);
   stove.increment();

   REQUIRE( stove.size() == 1 );

   // Increment the number of lives four more times for a total of five
   // increments. The last increment will be the end of the object's lifetime.
   for (unsigned int i = 0; i < 4; ++i)
   {
      stove.increment();
      REQUIRE( stove.size() == 1 );
   }

   // The sixth increment, which prunes the item from the stove.
   stove.increment();

   REQUIRE( stove.size() == 0 );
}

// Add a series of entries to one element of the stove. Verifies that values
// are shifted out of the history appropriately by checking that the average
// value changes over time.
TEST_CASE( "series of single graphs", "[stove]" )
{
   unsigned int num_constraints = 1;
   DynamicArray<stoveEdge_t> temp_graph(32);
   HeapMatrix temp_lagrange(num_constraints, 1);
   for (unsigned int i = 0; i < num_constraints; ++i)
   {
      stoveEdge_t temp_edge;
      temp_edge.nodeIdA = i;
      temp_edge.nodeIdB = 2 * i + 1;
      temp_edge.edgeId.constraintType = static_cast<oy::types::enumConstraint_t>(i % 7 + 1);
      temp_edge.edgeId.entity = 7;
      temp_edge.edgeId.offset = i + 123;

      std::cout << "type: " << static_cast<int>(temp_edge.edgeId.constraintType) << ", a: " << temp_edge.nodeIdA << ", b: " << temp_edge.nodeIdB << "\n";

      temp_graph.push_back(temp_edge);
      temp_lagrange(i, 0) = static_cast<float>(i);
   }

   oy::ConstraintStove stove(5);

   graph::types::subgraph_bookmark_t bookmark = {0, static_cast<unsigned int>(temp_graph.size())};

   graph::Subgraph<trecs::uid_t, oy::types::edgeId_t> subgraph(temp_graph, bookmark);

   stove.push(temp_lagrange, subgraph);

   temp_lagrange(0, 0) += 3.f;
   stove.push(temp_lagrange, subgraph);

   temp_lagrange(0, 0) += 3.f;
   stove.push(temp_lagrange, subgraph);
}

// Verify that the calculation of the key for the underlying map is correct
// in exactly one test case.
TEST_CASE( "verify standard key calculation", "[stove]" )
{
   oy::ConstraintStove stove(5);

   stoveEdge_t temp_edge;
   unsigned int i = 4;
   temp_edge.nodeIdA = i;
   temp_edge.nodeIdB = 2 * i + 1;
   temp_edge.edgeId.constraintType = static_cast<oy::types::enumConstraint_t>(i % 7 + 1);
   temp_edge.edgeId.entity = 2;
   temp_edge.edgeId.offset = 0;

   std::bitset<96> temp_hash = stove.calculateStandardKey(temp_edge);

   // The upper 32 bits are the constraint ID.
   REQUIRE(((temp_hash >> 64) == static_cast<int>(temp_edge.edgeId.constraintType)));

   // The middle 32 bits are body ID A.
   std::bitset<96> mask = temp_hash >> 64;
   mask <<= 64;
   temp_hash ^= mask;
   REQUIRE(((temp_hash >> 32) == temp_edge.nodeIdA));

   // The lowest 32 bits are body ID B.
   mask = temp_hash >> 32;
   mask <<= 32;
   temp_hash ^= mask;
   REQUIRE(((temp_hash) == temp_edge.nodeIdB));
}
