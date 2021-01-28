#include "graph.hpp"
#include "random_graph_utils.hpp"

#include <iostream>
#include <set>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#define SHOW_PRINT_STATEMENTS 0

void verify_edges_exist(
   DynamicArray<graph::types::edge_t<> > & edges,
   DynamicArray<graph::types::labeled_edge_t<> > & subgraph_edges
)
{
   for (unsigned int i = 0; i < edges.size(); ++i)
   {
      for (unsigned int j = 0; j < subgraph_edges.size(); ++j)
      {
         if (
            (edges[i].nodeIdA == subgraph_edges[j].nodeIdA) &&
            (edges[i].nodeIdB == subgraph_edges[j].nodeIdB) &&
            (edges[i].edgeId == subgraph_edges[j].edgeId)
         )
         {
            subgraph_edges.pop(j);
            break;
         }
      }
   }

   REQUIRE( subgraph_edges.size() == 0 );
}

TEST_CASE( "empty edge list connected component test", "[connectedComponent]" )
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   DynamicArray<int> markers(32);
   DynamicArray<graph::types::labeled_edge_t<> > labeled_edges(32);
   graph::connectedComponent(edges, markers, 0, 1, labeled_edges);

   REQUIRE( labeled_edges.size() == 0 );
}

TEST_CASE( "one-edge connected component test with sequential node IDs", "[connectedComponent]" )
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   edges.push_back({0, 1, 1, graph::types::enumTransitivity_t::TRANSITIVE});
   DynamicArray<int> markers(32);
   markers.push_back(0);
   markers.push_back(0);
   DynamicArray<graph::types::labeled_edge_t<> > labeled_edges(32);

   int subgraph_label = 1;
   graph::connectedComponent(edges, markers, 0, subgraph_label, labeled_edges);

   REQUIRE( labeled_edges.size() == 1 );
   REQUIRE( labeled_edges[0].label == subgraph_label );
   REQUIRE( labeled_edges[0].nodeIdA == edges[0].nodeIdA );
   REQUIRE( labeled_edges[0].nodeIdB == edges[0].nodeIdB );
}

TEST_CASE( "two-edge connected component test with sequential node IDs", "[connectedComponent]" )
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   edges.push_back({0, 1, 2, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({0, 2, 3, graph::types::enumTransitivity_t::TRANSITIVE});
   DynamicArray<int> markers(32);
   markers.push_back(0);
   markers.push_back(0);
   markers.push_back(0);
   DynamicArray<graph::types::labeled_edge_t<> > labeled_edges(32);

   int subgraph_label = 1;
   graph::connectedComponent(edges, markers, 0, subgraph_label, labeled_edges);

   REQUIRE( labeled_edges.size() == 2 );
   REQUIRE( labeled_edges[0].label == subgraph_label );

   REQUIRE( labeled_edges[1].label == subgraph_label );
}

// Generates one connected component out of a randomly assembled graph with a
// configurable number of edges per connected component. Verifies that the
// connected component identified by the graph algorithm is valid.
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

#if SHOW_PRINT_STATEMENTS
   std::cout << "original edge list\n";
   for (const auto edge : edges)
   {
      std::cout << "[" << edge.nodeIdA << ", " << edge.nodeIdB << ", " << edge.edgeType << ", " << edge.edgeId << "]\n";
   }
#endif

   DynamicArray<graph::types::edge_t<> > relabeled_edges(edges.size());
   DynamicArray<int> new_to_old_labels(edges.size());
   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);

#if SHOW_PRINT_STATEMENTS
   std::cout << "original edge list after relabel\n";
   for (const auto edge : edges)
   {
      std::cout << "[" << edge.nodeIdA << ", " << edge.nodeIdB << ", " << edge.edgeType << ", " << edge.edgeId << "]\n";
   }

   std::cout << "relabeled nodes original edge list\n";
   for (const auto edge : relabeled_edges)
   {
      std::cout << "[" << edge.nodeIdA << ", " << edge.nodeIdB << ", " << edge.edgeType << ", " << edge.edgeId << "]\n";
   }
#endif

   DynamicArray<int> markers(edges.size());
   for (unsigned int i = 0; i < new_to_old_labels.size(); ++i)
   {
      markers.push_back(0);
   }

   DynamicArray<graph::types::labeled_edge_t<> > relabeled_component_edges(edges.size());

   int subgraph_label = 1;
   graph::connectedComponent(relabeled_edges, markers, 0, subgraph_label, relabeled_component_edges);

#if SHOW_PRINT_STATEMENTS
   std::cout << "connected component edge list\n";
   for (const auto edge : relabeled_component_edges)
   {
      std::cout << "[" << edge.nodeIdA << ", " << edge.nodeIdB << ", " << edge.edgeId << "]\n";
   }
#endif

   REQUIRE( relabeled_component_edges.size() == num_edges_per_component );

   std::set<int> component_labels;

   for (const auto edge : relabeled_component_edges)
   {
      component_labels.insert(edge.label);
   }

   REQUIRE( component_labels.size() == 1 );

   // Take the subgraph and convert the nodes back to the old labels
   DynamicArray<graph::types::labeled_edge_t<> > rerelabeled_edges(32);
   for (const auto edge : relabeled_component_edges)
   {
      rerelabeled_edges.push_back(
         {
            new_to_old_labels[edge.nodeIdA],
            new_to_old_labels[edge.nodeIdB],
            edge.edgeId,
            edge.label
         }
      );
   }

   verify_edges_exist(edges, rerelabeled_edges);
}

TEST_CASE( "10-edge connected component test with random node IDs", "[connectedComponent]" )
{
   test_parameterized_random_graph(2, 5);
}

TEST_CASE( "20-edge connected component test with random node IDs [4,5]", "[connectedComponent]" )
{
   test_parameterized_random_graph(4, 5);
}

TEST_CASE( "20-edge connected component test with random node IDs [5,4]", "[connectedComponent]" )
{
   std::cout << "test 20 5, 4\n";
   test_parameterized_random_graph(5, 4);
}

TEST_CASE( "100-edge connected component test with random node IDs [10,10]", "[connectedComponent]" )
{
   test_parameterized_random_graph(10, 10);
}

TEST_CASE( "100-edge connected component test with random node IDs [5,20]", "[connectedComponent]" )
{
   test_parameterized_random_graph(5, 20);
}

TEST_CASE( "100-edge connected component test with random node IDs [4,25]", "[connectedComponent]" )
{
   std::cout << "100 4, 25\n";
   test_parameterized_random_graph(4, 25);
}

TEST_CASE( "100-edge connected component test with random node IDs [25,4]", "[connectedComponent]" )
{
   test_parameterized_random_graph(25, 4);
}

TEST_CASE( "run connected component on a bunch of different configurations", "[connectedComponent]" )
{
   for (int i = 0; i < 500; ++i)
   {
      int num_components = edbdmath::random_int(1, 50);
      int num_edges_per_component = edbdmath::random_int(1, 50);
#if SHOW_PRINT_STATEMENTS
      std::cout << "num components: " << num_components << ", num edges per component: " << num_edges_per_component << "\n";
#endif
      test_parameterized_random_graph(num_components, num_edges_per_component);
   }
}

TEST_CASE( "small terminal node test", "[connectedComponent]")
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   edges.push_back({0, 1, 2, graph::types::enumTransitivity_t::NODE_A_TERMINAL});
   edges.push_back({0, 2, 12, graph::types::enumTransitivity_t::NODE_A_TERMINAL});
   edges.push_back({3, 2, 21, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({3, 4, 4, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({1, 5, 9, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({6, 5, 7, graph::types::enumTransitivity_t::TRANSITIVE});

   DynamicArray<int> new_to_old_labels(32);
   DynamicArray<graph::types::edge_t<> > relabeled_edges(32);

   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);
   DynamicArray<int> markers(32);
   for (unsigned int i = 0; i < new_to_old_labels.size(); ++i)
   {
      markers.push_back(0);
   }

#if SHOW_PRINT_STATEMENTS
   std::cout << "original edge list\n";
   for (const auto edge : edges)
   {
      std::cout << "[" << edge.nodeIdA << ", " << edge.nodeIdB << ", " << edge.edgeType << ", " << edge.edgeId << "]\n";
   }
#endif

   DynamicArray<graph::types::labeled_edge_t<> > labeled_edges(32);

   int subgraph_label = 1;
   graph::connectedComponent(relabeled_edges, markers, 0, subgraph_label, labeled_edges);

#if SHOW_PRINT_STATEMENTS
   std::cout << "component\n";
   for (const auto edge : labeled_edges)
   {
      std::cout << "\t[" << edge.nodeIdA << ", " << edge.nodeIdB << ", " << edge.edgeId << "]\n";
   }
#endif

   REQUIRE( labeled_edges.size() == 3 );
   REQUIRE( labeled_edges[0].label == subgraph_label );
   REQUIRE( labeled_edges[1].label == subgraph_label );
   REQUIRE( labeled_edges[2].label == subgraph_label );

   // Take the subgraph and convert the nodes back to the old labels
   DynamicArray<graph::types::labeled_edge_t<> > rerelabeled_edges(32);
   for (const auto edge : labeled_edges)
   {
      rerelabeled_edges.push_back(
         {
            new_to_old_labels[edge.nodeIdA],
            new_to_old_labels[edge.nodeIdB],
            edge.edgeId,
            edge.label
         }
      );
   }

   verify_edges_exist(edges, rerelabeled_edges);
}

// This isn't a valid test for the connected component calculator or the
// connected component generator, really. But it *does* verify that connected
// component calculator and the connected component generator both agree that
// the connected component generator only generates a graph with one connected
// component. It's certainly possible that they're both wrong, but this test
// at least gives me some feel-good vibes.
TEST_CASE( "multiple connected components", "[connectedComponent]" )
{
   for (int i = 3; i < 100; ++i)
   {
      DynamicArray<graph::types::edge_t<> > edges(i);
      test_utils::randomConnectedComponent(i, edges);

      DynamicArray<graph::types::edge_t<> > relabeled_node_edges(edges.size());
      DynamicArray<int> new_to_old_node_labels(edges.size() * 2);
      graph::relabelNodesSequentially(edges, new_to_old_node_labels, relabeled_node_edges);

      DynamicArray<int> visited_nodes(new_to_old_node_labels.size(), 0);

      DynamicArray<graph::types::labeled_edge_t<> > labeled_edges(edges.size());
      graph::connectedComponent(relabeled_node_edges, visited_nodes, relabeled_node_edges[0].nodeIdA, 1, labeled_edges);

      REQUIRE( labeled_edges.size() == edges.size() );
   }
}
