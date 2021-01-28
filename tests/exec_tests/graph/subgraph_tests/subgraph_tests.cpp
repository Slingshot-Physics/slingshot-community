#include "graph.hpp"
#include "subgraph.hpp"
#include "random_graph_utils.hpp"

#include <iostream>
#include <cstdint>
#include <set>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#define ENABLE_PRINT_STATEMENTS 0

// Loops over all of the connected components from a decomposed graph, builds
// a subgraph out of each connected component, and verifies that the subgraph
// edge list has the same fields as the bookmarked connected component in the
// original decomposed edge list.
template <typename Node_T, typename EdgeId_T>
void test_subgraph_against_components(
   DynamicArray<graph::types::labeled_edge_t<Node_T, EdgeId_T> > & labeled_edges,
   DynamicArray<graph::types::subgraph_bookmark_t> & bookmarks
)
{
   for (unsigned int j = 0; j < bookmarks.size(); ++j)
   {
      // Grab a subgraph for each bookmark in the decomposed graph.
      graph::Subgraph<Node_T, EdgeId_T> sub(labeled_edges, bookmarks[j]);

      REQUIRE( sub.size() == bookmarks[j].size );

      // Test that the subgraph has all the correct components using the
      // random access operator.
      for (unsigned int i = 0; i < sub.size(); ++i)
      {
         REQUIRE( sub.at(i).edgeId == labeled_edges.at(i + bookmarks[j].startIndex).edgeId );
         REQUIRE( sub.at(i).label == labeled_edges.at(i + bookmarks[j].startIndex).label );
         REQUIRE( sub.at(i).nodeIdA == labeled_edges.at(i + bookmarks[j].startIndex).nodeIdA );
         REQUIRE( sub.at(i).nodeIdB == labeled_edges.at(i + bookmarks[j].startIndex).nodeIdB );
      }

      // Test that the subgraph has all the correct components using range-
      // based for loops.
      {
         unsigned int i = 0;
         for (const auto & edge : sub)
         {
            REQUIRE( edge.edgeId == labeled_edges.at(i + bookmarks[j].startIndex).edgeId );
            REQUIRE( edge.label == labeled_edges.at(i + bookmarks[j].startIndex).label );
            REQUIRE( edge.nodeIdA == labeled_edges.at(i + bookmarks[j].startIndex).nodeIdA );
            REQUIRE( edge.nodeIdB == labeled_edges.at(i + bookmarks[j].startIndex).nodeIdB );
            REQUIRE( i < bookmarks[j].size );
            ++i;
         }
      }
   }
}

TEST_CASE( "small graph bookmarks", "[connected_components]" )
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   edges.push_back({0, 1, 0, graph::types::enumTransitivity_t::NODE_A_TERMINAL});
   edges.push_back({0, 2, 3, graph::types::enumTransitivity_t::NODE_A_TERMINAL});
   edges.push_back({3, 2, 9, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({3, 4, 10, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({1, 5, 11, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({6, 5, 13, graph::types::enumTransitivity_t::TRANSITIVE});

   DynamicArray<graph::types::labeled_edge_t<> > labeled_edges(32);

   graph::connectedComponentsDecomposition(edges, labeled_edges);

   DynamicArray<graph::types::subgraph_bookmark_t> bookmarks(32);

#if ENABLE_PRINT_STATEMENTS
   std::cout << "connected components\n";
   for (unsigned int i = 0; i < labeled_edges.size(); ++i)
   {
      std::cout << "\t" << labeled_edges[i].nodeIdA << ", " << labeled_edges[i].nodeIdB << " " << labeled_edges[i].label << "\n";
   }
#endif

   graph::calculateSubgraphBookmarks(labeled_edges, bookmarks);

   REQUIRE(bookmarks.size() == 2);
   REQUIRE(bookmarks[0].startIndex == 0);
   REQUIRE(bookmarks[0].size == 3);
   REQUIRE(bookmarks[1].startIndex == 3);
   REQUIRE(bookmarks[1].size == 3);

   test_subgraph_against_components(labeled_edges, bookmarks);
}

TEST_CASE( "small uint64_t connected components", "[connected_components]" )
{
   DynamicArray<graph::types::edge_t<uint64_t> > edges(32);
   edges.push_back({0, 1, 0, graph::types::enumTransitivity_t::NODE_A_TERMINAL});
   edges.push_back({0, 2, 1, graph::types::enumTransitivity_t::NODE_A_TERMINAL});
   edges.push_back({3, 2, 3, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({3, 4, 4, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({1, 5, 7, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({6, 5, 10, graph::types::enumTransitivity_t::TRANSITIVE});

   DynamicArray<graph::types::labeled_edge_t<uint64_t> > labeled_edges(32);

   graph::connectedComponentsDecomposition(edges, labeled_edges);

   DynamicArray<graph::types::subgraph_bookmark_t> bookmarks(32);

#if ENABLE_PRINT_STATEMENTS
   std::cout << "connected components\n";
   for (unsigned int i = 0; i < labeled_edges.size(); ++i)
   {
      std::cout << "\t" << labeled_edges[i].nodeIdA << ", " << labeled_edges[i].nodeIdB << " " << labeled_edges[i].label << "\n";
   }
#endif

   graph::calculateSubgraphBookmarks(labeled_edges, bookmarks);

   REQUIRE(bookmarks.size() == 2);
   REQUIRE(bookmarks[0].startIndex == 0);
   REQUIRE(bookmarks[0].size == 3);
   REQUIRE(bookmarks[1].startIndex == 3);
   REQUIRE(bookmarks[1].size == 3);

   test_subgraph_against_components(labeled_edges, bookmarks);
}

// This is for generating plottable graphs.
TEST_CASE( "random graph with terminations" )
{
   DynamicArray<graph::types::edge_t<> > edges(50 * 20);

   test_utils::randomNumComponentsGraphWithTerminations(
      50,
      20,
      5,
      edges
   );

   DynamicArray<int> old_to_new_labels(edges.size());

   DynamicArray<graph::types::edge_t<> > new_edges(edges.size());

   graph::relabelNodesSequentially(edges, old_to_new_labels, new_edges);

#if ENABLE_PRINT_STATEMENTS
   std::cout << "graph with terminations edges:\n";
   for (const auto edge : new_edges)
   {
      std::cout << "[" << edge.nodeIdA << "," << edge.nodeIdB << "],\n";
   }
#endif
}
