#include "random_graph_utils.hpp"

#include <iostream>
#include <set>

int dummy = edbdmath::seed_rng_ret();

namespace test_utils
{
   graph::types::edge_t<> randomEdge(int node_id_min, int node_id_max, int edge_id)
   {
      if (node_id_min == node_id_max)
      {
         std::cout << "you called the random edge generator with the same min/max node UIDs, did you mean to?\n";
         std::cout << node_id_min << ", " << node_id_max << "\n";
         return {-1, -1, edge_id, graph::types::enumTransitivity_t::TRANSITIVE};
      }

      int node_id_a = edbdmath::random_int(node_id_min, node_id_max);
      int node_id_b = node_id_a;
      do
      {
         node_id_b = edbdmath::random_int(node_id_min, node_id_max);
      } while (node_id_a == node_id_b);

      graph::types::edge_t<> temp_edge = {
         node_id_a, node_id_b, edge_id, graph::types::enumTransitivity_t::TRANSITIVE
      };

      return temp_edge;
   }

   void randomConnectedComponent(
      int num_edges, DynamicArray<graph::types::edge_t<> > & edges
   )
   {
      edges.clear();
      std::set<int> nodes;

      for (int i = 0; i < num_edges; ++i)
      {
         graph::types::edge_t<> temp_edge;
         temp_edge.edgeId = i;
         temp_edge.edgeType = graph::types::enumTransitivity_t::TRANSITIVE;
         if (i == 0)
         {
            temp_edge.nodeIdA = edbdmath::random_int(0, num_edges);
            do
            {
               temp_edge.nodeIdB = edbdmath::random_int(0, num_edges);
            } while (temp_edge.nodeIdA == temp_edge.nodeIdB);
            // temp_edge.edgeType = graph::types::enumTransitivity_t::TRANSITIVE;
         }
         else
         {
            int random_index = edbdmath::random_int(0, nodes.size() - 1);

            int j = 0;
            for (const auto node : nodes)
            {
               if (j == random_index)
               {
                  temp_edge.nodeIdA = node;
                  break;
               }
               ++j;
            }

            do
            {
               temp_edge.nodeIdB = edbdmath::random_int(0, num_edges);
            } while (temp_edge.nodeIdA == temp_edge.nodeIdB);
         }
         nodes.insert(temp_edge.nodeIdA);
         nodes.insert(temp_edge.nodeIdB);
         // std::cout << "temp edge: " << temp_edge.nodeIdA << ", " << temp_edge.nodeIdB << "\n";
         edges.push_back(temp_edge);
      }
   }

   void randomConnectedComponent(
      int num_edges,
      int max_terminal_nodes,
      DynamicArray<graph::types::edge_t<> > & edges
   )
   {
      if (max_terminal_nodes >= num_edges)
      {
         std::cout << "Too many terminal nodes requested, reducing " << max_terminal_nodes << " to " << (num_edges / 2) << "\n";
         max_terminal_nodes = num_edges / 2;
      }

      int num_terminal_nodes = 0;

      edges.clear();
      std::set<int> nodes;

      for (int i = 0; i < num_edges; ++i)
      {
         graph::types::edge_t<> temp_edge;
         temp_edge.edgeId = i;
         temp_edge.edgeType = graph::types::enumTransitivity_t::TRANSITIVE;
         if (i == 0)
         {
            temp_edge.nodeIdA = edbdmath::random_int(0, num_edges);
            do
            {
               temp_edge.nodeIdB = edbdmath::random_int(0, num_edges);
            } while (temp_edge.nodeIdA == temp_edge.nodeIdB);
            // temp_edge.edgeType = graph::types::enumTransitivity_t::TRANSITIVE;
         }
         else
         {
            {
               int random_index = edbdmath::random_int(0, nodes.size() - 1);

               int j = 0;
               for (const auto node : nodes)
               {
                  if (j == random_index)
                  {
                     temp_edge.nodeIdA = node;
                     break;
                  }
                  ++j;
               }
            }

            // Load node ID B with a default node value. Add a new node 10%
            // of the time. Connected the edge to a node that's already in use
            // the other 90% of the time.
            if (edbdmath::random_float() < 0.1f)
            {
               do
               {
                  temp_edge.nodeIdB = edbdmath::random_int(0, num_edges);
               } while (temp_edge.nodeIdA == temp_edge.nodeIdB);
            }
            else
            {
               int random_index = edbdmath::random_int(0, nodes.size() - 1);
               int j = 0;
               for (const auto node : nodes)
               {
                  if (j == random_index)
                  {
                     temp_edge.nodeIdB = node;
                     break;
                  }
                  ++j;
               }
            }

            // Override the default node value with a terminal node only if
            // the maximum number of terminal nodes hasn't been reached and if
            // a coin toss pans out.
            if (
               (num_terminal_nodes < max_terminal_nodes) &&
               (edbdmath::random_float() < 0.5f)
            )
            {
               temp_edge.nodeIdB = -1 * i;
               temp_edge.edgeType = graph::types::enumTransitivity_t::NODE_B_TERMINAL;
               ++num_terminal_nodes;
            }
         }
         nodes.insert(temp_edge.nodeIdA);
         if (temp_edge.edgeType != graph::types::enumTransitivity_t::NODE_B_TERMINAL)
         {
            nodes.insert(temp_edge.nodeIdB);
         }
         // std::cout << "temp edge: " << temp_edge.nodeIdA << ", " << temp_edge.nodeIdB << "\n";
         edges.push_back(temp_edge);
      }

      // std::cout << "connected component with " << num_terminal_nodes << " terminations\n";
      // for (const auto edge: edges)
      // {
      //    std::cout << edge.nodeIdA << "-" << edge.nodeIdB << "\n";
      // }
   }

   void randomNumComponentsGraph(
      int num_components,
      int num_edges_per_component,
      DynamicArray<graph::types::edge_t<> > & edges
   )
   {
      // std::cout << "num components: " << num_components << "\n";
      // std::cout << "num edges per component: " << num_edges_per_component << "\n";
      edges.clear();

      // Used to provide daylight between node IDs in connected components
      int component_node_id_offset = num_edges_per_component + 2;

      for (int i = 0; i < num_components; ++i)
      {
         DynamicArray<graph::types::edge_t<> > subgraph(num_edges_per_component);
         randomConnectedComponent(num_edges_per_component, subgraph);

         for (auto & edge : subgraph)
         {
            edge.nodeIdA += component_node_id_offset * i;
            edge.nodeIdB += component_node_id_offset * i;
            edge.edgeId += component_node_id_offset * i;

            // std::cout << "\tpushing back: " << edge.nodeIdA << ", " << edge.nodeIdB << "\n";
            edges.push_back(edge);
         }
      }

      // std::cout << "edge list before shuffling:\n";
      // for (unsigned int i = 0; i < edges.size(); ++i)
      // {
      //    std::cout << "[" << edges[i].nodeIdA << ", " << edges[i].nodeIdB << ", " << edges[i].edgeType << ", " << edges[i].edgeId << "]\n";
      // }

      shuffleEdgeList(edges);
   }

   void randomNumComponentsGraphWithTerminations(
      int num_components,
      int num_edges_per_component,
      int max_terminal_nodes_per_component,
      DynamicArray<graph::types::edge_t<> > & edges
   )
   {
      edges.clear();

      // Used to provide daylight between node IDs in connected components
      int component_node_id_offset = num_edges_per_component + 2;

      for (int i = 0; i < num_components; ++i)
      {
         DynamicArray<graph::types::edge_t<> > subgraph(
            num_edges_per_component
         );
         randomConnectedComponent(
            num_edges_per_component, max_terminal_nodes_per_component, subgraph
         );

         for (auto & edge : subgraph)
         {
            edge.nodeIdA += component_node_id_offset * i;
            if (edge.edgeType == graph::types::enumTransitivity_t::NODE_B_TERMINAL)
            {
               edge.nodeIdB -= component_node_id_offset * i;
            }
            else
            {
               edge.nodeIdB += component_node_id_offset * i;
            }
            edge.edgeId += component_node_id_offset * i;

            edges.push_back(edge);
         }
      }

      shuffleEdgeList(edges);
   }

   void shuffleEdgeList(DynamicArray<graph::types::edge_t<> > & edges)
   {
      unsigned int num_edges = edges.size();
      for (unsigned int i = 0; i < num_edges; ++i)
      {
         int swap_index = edbdmath::random_int(0, num_edges - 1);
         const auto temp_edge = edges[i];

         edges[i] = edges[swap_index];
         edges[swap_index] = temp_edge;
      }
   }
}
