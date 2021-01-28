#include "graph.hpp"
#include "random_graph_utils.hpp"

#include <iostream>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

void verify_edges_reconstructed(
   DynamicArray<graph::types::edge_t<> > & edges,
   DynamicArray<int> & new_to_old_labels,
   DynamicArray<graph::types::edge_t<> > & relabeled_edges
)
{
   DynamicArray<graph::types::edge_t<> > rerelabeled_edges(relabeled_edges.size());

   for (const auto edge : relabeled_edges)
   {
      rerelabeled_edges.push_back(
         {
            new_to_old_labels[edge.nodeIdA],
            new_to_old_labels[edge.nodeIdB],
            edge.edgeId,
            graph::types::enumTransitivity_t::TRANSITIVE
         }
      );
   }

   REQUIRE( edges.size() == relabeled_edges.size() );

   for (unsigned int i = 0; i < edges.size(); ++i)
   {
      for (unsigned int j = 0; j < rerelabeled_edges.size(); ++j)
      {
         if (
            (edges[i].nodeIdA == rerelabeled_edges[j].nodeIdA) &&
            (edges[i].nodeIdB == rerelabeled_edges[j].nodeIdB) &&
            (edges[i].edgeId == rerelabeled_edges[j].edgeId)
         )
         {
            rerelabeled_edges.pop(j);
            break;
         }
      }
   }

   REQUIRE( rerelabeled_edges.size() == 0 );
}

// Verifies that an empty edge list generates a zero-size relabeled edge list.
TEST_CASE( "empty relabeling test", "[sequential_relabel]" )
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   DynamicArray<int> new_to_old_labels(32);
   DynamicArray<graph::types::edge_t<> > relabeled_edges(32);

   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);

   REQUIRE(new_to_old_labels.size() == 0);
   REQUIRE(relabeled_edges.size() == 0);
}

// Verifies that relabeling produces sequential node IDs.
TEST_CASE( "small relabeling test", "[sequential_relabel]" )
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   edges.push_back({-5, -6, 0, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({16, -6, 1, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({-10000, 0, 2, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({1009, 0, 3, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({1009, 33, 4, graph::types::enumTransitivity_t::TRANSITIVE});
   DynamicArray<int> new_to_old_labels(32);
   DynamicArray<graph::types::edge_t<> > relabeled_edges(32);

   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);

   REQUIRE(relabeled_edges.size() == edges.size());

   verify_edges_reconstructed(edges, new_to_old_labels, relabeled_edges);
}

// Generates hard-coded edges in an edge list, performs the sequential
// relabeling of the nodes in the edge list, and verifies that the relabeling
// reproduces the original edge list.
TEST_CASE( "node IDs already sequential", "[sequential_relabel]" )
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   edges.push_back({0, 1, 0, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({2, 1, 1, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({2, 3, 2, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({3, 4, 3, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({5, 4, 4, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({5, 6, 5, graph::types::enumTransitivity_t::TRANSITIVE});
   DynamicArray<int> new_to_old_labels(32);
   DynamicArray<graph::types::edge_t<> > relabeled_edges(32);

   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);

   REQUIRE(relabeled_edges.size() == edges.size());

   verify_edges_reconstructed(edges, new_to_old_labels, relabeled_edges);
}

// Generates 2^18 edges with random node IDs in the range [-2^18, 2^18],
// performs the sequential relabeling of the nodes in the edge list, and
// verifies that the relabeling reproduces the original edge list.
TEST_CASE( "big random edge list", "[sequential_relabel]" )
{
   const int max_num_edges = (1 << 18);
   DynamicArray<graph::types::edge_t<> > edges(max_num_edges);
   for (unsigned int i = 0; i < max_num_edges; ++i)
   {
      edges.push_back(test_utils::randomEdge(-1 * max_num_edges, max_num_edges));
   }
   DynamicArray<int> new_to_old_labels(32);
   DynamicArray<graph::types::edge_t<> > relabeled_edges(32);

   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);

   REQUIRE(relabeled_edges.size() == edges.size());

   for (unsigned int i = 0; i < edges.size(); ++i)
   {
      REQUIRE( new_to_old_labels[relabeled_edges[i].nodeIdA] == edges[i].nodeIdA);
      REQUIRE( new_to_old_labels[relabeled_edges[i].nodeIdB] == edges[i].nodeIdB);
   }
}

TEST_CASE( "relabel small terminable edge list", "[sequential_relabel]")
{
   DynamicArray<graph::types::edge_t<> > edges(6);
   edges.push_back({0, 1, 3, graph::types::enumTransitivity_t::NODE_A_TERMINAL});
   edges.push_back({0, 2, 4, graph::types::enumTransitivity_t::NODE_A_TERMINAL});
   edges.push_back({2, 3, 7, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({4, 3, 18, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({1, 6, 5, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({7, 6, 2, graph::types::enumTransitivity_t::TRANSITIVE});

   DynamicArray<int> new_to_old_labels(6);
   DynamicArray<graph::types::edge_t<> > relabeled_edges(6);

   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);

   std::cout << "relabeled edges with terminal nodes\n";
   for (const auto edge : relabeled_edges)
   {
      std::cout << "\t[" << edge.nodeIdA << ", " << edge.nodeIdB << "]\n";
   }

   std::cout << "reconstructed edges\n";
   for (const auto edge : relabeled_edges)
   {
      std::cout << "\t[" << new_to_old_labels[edge.nodeIdA] << ", " << new_to_old_labels[edge.nodeIdB ] << "]\n";
   }

   REQUIRE( relabeled_edges.size() == edges.size() );

   verify_edges_reconstructed(edges, new_to_old_labels, relabeled_edges);
}

TEST_CASE( "relabel medium terminable edge list", "[sequential_relabel]")
{
   DynamicArray<graph::types::edge_t<> > edges(256);
   test_utils::randomNumComponentsGraph(16, 16, edges);

   for (unsigned int i = 0; i < 256; i += 16)
   {
      edges[i].nodeIdA = -3;
      edges[i].edgeType = graph::types::enumTransitivity_t::NODE_A_TERMINAL;
   }

   for (unsigned int i = 2; i < 256; i += 16)
   {
      edges[i].nodeIdB = -7;
      edges[i].edgeType = graph::types::enumTransitivity_t::NODE_B_TERMINAL;
   }

   DynamicArray<int> new_to_old_labels(256);
   DynamicArray<graph::types::edge_t<> > relabeled_edges(256);

   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);

   std::cout << "relabeled edges with terminal nodes\n";
   for (const auto edge : relabeled_edges)
   {
      std::cout << "\t[" << edge.nodeIdA << ", " << edge.nodeIdB << "]\n";
   }

   std::cout << "reconstructed edges\n";
   for (const auto edge : relabeled_edges)
   {
      std::cout << "\t[" << new_to_old_labels[edge.nodeIdA] << ", " << new_to_old_labels[edge.nodeIdB ] << "]\n";
   }

   REQUIRE( relabeled_edges.size() == edges.size() );

   verify_edges_reconstructed(edges, new_to_old_labels, relabeled_edges);
}

TEST_CASE( "small relabeling test with edge IDs", "[sequential_relabel]" )
{
   DynamicArray<graph::types::edge_t<> > edges(32);
   edges.push_back({-5, -6, 0, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({16, -6, 2, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({-10000, 0, 3, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({1009, 0, 7, graph::types::enumTransitivity_t::TRANSITIVE});
   edges.push_back({1009, 33, 8, graph::types::enumTransitivity_t::TRANSITIVE});
   DynamicArray<int> new_to_old_labels(32);
   DynamicArray<graph::types::edge_t<> > relabeled_edges(32);

   graph::relabelNodesSequentially(edges, new_to_old_labels, relabeled_edges);

   REQUIRE(relabeled_edges.size() == edges.size());

   verify_edges_reconstructed(edges, new_to_old_labels, relabeled_edges);
}
