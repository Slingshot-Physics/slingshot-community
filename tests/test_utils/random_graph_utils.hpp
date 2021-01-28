#ifndef RANDOM_GRAPH_UTILS_HEADER
#define RANDOM_GRAPH_UTILS_HEADER

#include "graph.hpp"
#include "random_utils.hpp"

namespace test_utils
{
   graph::types::edge_t<> randomEdge(int node_id_min, int node_id_max, int edge_id=0);

   // Generates one graph with one connected component and a specified number
   // of edges.
   void randomConnectedComponent(
      int num_edges, DynamicArray<graph::types::edge_t<> > & edges
   );

   // Generates one graph with one connected component, a specified number of
   // edges, and a maximum number edges with terminal nodes.
   void randomConnectedComponent(
      int num_edges,
      int max_terminal_nodes,
      DynamicArray<graph::types::edge_t<> > & edges
   );

   // Generates an edge list with a guaranteed number of connected components
   // and number of edges per connected component.
   // The edge IDs for each node are unique.
   void randomNumComponentsGraph(
      int num_components,
      int num_edges_per_component,
      DynamicArray<graph::types::edge_t<> > & edges
   );

   // Generates an edge list with:
   //   - A guaranteed number of connected components
   //   - A guaranteed number of edges per connected component
   //   - A maximum number of edges with terminal nodes per component
   // The edge IDs for each node are unique.
   void randomNumComponentsGraphWithTerminations(
      int num_components,
      int num_edges_per_component,
      int max_terminal_nodes_per_component,
      DynamicArray<graph::types::edge_t<> > & edges
   );

   // Randomly swaps the positions of edges in an edge list.
   void shuffleEdgeList(DynamicArray<graph::types::edge_t<> > & edges);
}

#endif
