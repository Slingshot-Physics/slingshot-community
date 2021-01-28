#include "graph.hpp"
#include "random_graph_utils.hpp"

#include <iostream>
#include <cstdint>
#include <set>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#define ENABLE_PRINT_STATEMENTS 0

void test_parameterized_random_graph(
   int num_components, int num_edges_per_component
)
{
   DynamicArray<graph::types::edge_t<> > edges(
      num_components * num_edges_per_component
   );

   test_utils::randomNumComponentsGraph(
      num_components, num_edges_per_component, edges
   );

#if ENABLE_PRINT_STATEMENTS
   std::cout << "generated the graph\n";

   for (const auto edge : edges)
   {
      std::cout << "\t[" << edge.nodeIdA << ", " << edge.nodeIdB << "]\n";
   }
#endif

   // The dynamic array of connected components
   DynamicArray<graph::types::labeled_edge_t<> > labeled_edges(
      num_components * num_edges_per_component
   );

   graph::connectedComponentsDecomposition(edges, labeled_edges);

#if ENABLE_PRINT_STATEMENTS
   std::cout << "labeled graph: \n";

   for (const auto edge : labeled_edges)
   {
      std::cout << "\t" << edge.label << " [" << edge.nodeIdA << ", " << edge.nodeIdB << "]\n";
   }
#endif

   std::set<int> component_labels;

   for (const auto edge : labeled_edges)
   {
      component_labels.insert(edge.label);
   }

   // Verify that the number of connected components that were found matches
   // the prescribed number of connected components.
   REQUIRE( component_labels.size() == num_components );

   // Verify that all of the labeled edges have edge ID's that originated from
   // the original edge list.
   for (const auto edge : edges)
   {
      for (unsigned int j = 0; j < labeled_edges.size(); ++j)
      {
         if (edge.edgeId == labeled_edges[j].edgeId)
         {
            labeled_edges.pop(j);
            break;
         }
      }
   }

   REQUIRE( labeled_edges.size() == 0 );
}

void test_parameterized_random_graph_with_terminations(
   int num_components,
   int num_edges_per_component,
   int max_terminations_per_component
)
{
   DynamicArray<graph::types::edge_t<> > edges(
      num_components * num_edges_per_component
   );
   test_utils::randomNumComponentsGraphWithTerminations(
      num_components,
      num_edges_per_component,
      max_terminations_per_component,
      edges
   );

#if ENABLE_PRINT_STATEMENTS
   std::cout << "generated the graph\n";
   for (const auto edge : edges)
   {
      std::cout << "\t[" << edge.nodeIdA << ", " << edge.nodeIdB << "]\n";
   }
#endif

   DynamicArray<graph::types::labeled_edge_t<> > labeled_edges(
      num_components * num_edges_per_component
   );
   graph::connectedComponentsDecomposition(edges, labeled_edges);

   std::map<int, int> subgraph_label_counts;

   for (const auto edge : labeled_edges)
   {
      if (subgraph_label_counts.find(edge.label) != subgraph_label_counts.end())
      {
         subgraph_label_counts[edge.label] += 1;
      }
      else
      {
         subgraph_label_counts[edge.label] = 1;
      }
   }

   std::set<int> component_labels;

   for (const auto edge : labeled_edges)
   {
      component_labels.insert(edge.label);
   }

   // Verify that the number of connected components that were found matches
   // the prescribed number of connected components.
   REQUIRE( component_labels.size() == num_components );

   // Verify that all of the labeled edges have edge ID's that originated from
   // the original edge list.
   for (const auto edge : edges)
   {
      for (unsigned int j = 0; j < labeled_edges.size(); ++j)
      {
         if (edge.edgeId == labeled_edges[j].edgeId)
         {
            labeled_edges.pop(j);
            break;
         }
      }
   }

   REQUIRE( labeled_edges.size() == 0 );
}

TEST_CASE( "2,5", "[connected_components]" )
{
   test_parameterized_random_graph(2, 5);
}

TEST_CASE( "4,5", "[connected_components]" )
{
   test_parameterized_random_graph(4, 5);
}

TEST_CASE( "3,7", "[connected_components]" )
{
   test_parameterized_random_graph(3, 7);
}

TEST_CASE( "7,4", "[connected_components]" )
{
   test_parameterized_random_graph(7, 4);
}

TEST_CASE( "1000,22", "[connected_components]" )
{
   test_parameterized_random_graph(1000, 22);
}

TEST_CASE( "3,7,1 with terminations", "[connected_components]" )
{
   test_parameterized_random_graph_with_terminations(3, 7, 1);
}

TEST_CASE( "21,15,5 with terminations", "[connected_components]" )
{
   test_parameterized_random_graph_with_terminations(21, 15, 5);
}

TEST_CASE( "100,22,10 with terminations", "[connected_components]" )
{
   test_parameterized_random_graph_with_terminations(100, 22, 10);
}

TEST_CASE( "3,7,2 with terminations", "[connected_components]" )
{
   test_parameterized_random_graph_with_terminations(3, 7, 2);
}

TEST_CASE( "5,60,30 with terminations", "[connected_components]" )
{
   test_parameterized_random_graph_with_terminations(5, 60, 30);
}

TEST_CASE( "22,100,75 with terminations", "[connected_components]" )
{
   test_parameterized_random_graph_with_terminations(22, 100, 75);
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
