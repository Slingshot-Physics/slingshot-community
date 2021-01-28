#ifndef GRAPH_HEADER
#define GRAPH_HEADER

#include "graph_types.hpp"

#include "dynamic_array.hpp"

#include <algorithm>
#include <map>
#include <set>

#include <iostream>

namespace graph
{
   struct reverseIndex_t
   {
      inline bool operator()(const int a, const int b) const
      {
         return a >= b;
      }
   };

   template <typename Node_T, typename EdgeId_T>
   void relabelNodesSequentially(
      const DynamicArray<types::edge_t<Node_T, EdgeId_T> > & old_edges,
      DynamicArray<Node_T> & new_to_old_labels,
      DynamicArray<types::edge_t<int, EdgeId_T> > & new_edges
   )
   {
      // Get the set of nodes that are transitive.
      std::set<Node_T> old_transitive_node_labels;

      for (const auto & edge : old_edges)
      {
         if (edge.edgeType != types::enumTransitivity_t::NODE_A_TERMINAL)
         {
            old_transitive_node_labels.insert(edge.nodeIdA);
         }
         if (edge.edgeType != types::enumTransitivity_t::NODE_B_TERMINAL)
         {
            old_transitive_node_labels.insert(edge.nodeIdB);
         }
      }

      // Build a map of the old transitive node types to the new transitive
      // node types.
      std::map<Node_T, int> transitive_old_to_new;

      int i = 0;
      for (auto & node_uid : old_transitive_node_labels)
      {
         new_to_old_labels.push_back(node_uid);
         transitive_old_to_new[node_uid] = i;
         ++i;
      }

      // Loop over all of the old edges, add labels mapping old terminal nodes
      // to ints. Build new edges out of the mapping from old nodes to new
      // nodes.
      new_edges.clear();
      for (const auto & old_edge : old_edges)
      {
         types::edge_t<int, EdgeId_T> new_edge;

         switch(old_edge.edgeType)
         {
            case types::enumTransitivity_t::NODE_A_TERMINAL:
               new_to_old_labels.push_back(old_edge.nodeIdA);
               new_edge.nodeIdA = static_cast<int>(new_to_old_labels.size()) - 1;
               new_edge.nodeIdB = transitive_old_to_new[old_edge.nodeIdB];
               break;
            case types::enumTransitivity_t::NODE_B_TERMINAL:
               new_to_old_labels.push_back(old_edge.nodeIdB);
               new_edge.nodeIdA = transitive_old_to_new[old_edge.nodeIdA];
               new_edge.nodeIdB = static_cast<int>(new_to_old_labels.size()) - 1;
               break;
            case types::enumTransitivity_t::TRANSITIVE:
               new_edge.nodeIdA = transitive_old_to_new[old_edge.nodeIdA];
               new_edge.nodeIdB = transitive_old_to_new[old_edge.nodeIdB];
               break;
         }

         new_edge.edgeId = old_edge.edgeId;
         new_edges.push_back(new_edge);
      }
   }

   // This function only operates on edge lists whose node identifiers are
   // sequential integers. Edge flags are ignored (or the function assumes all
   // edges are transitive).
   //
   // It is recommended that the user call 'relabelNodesSequentially' on an
   // edge list with flags.
   //
   // Performs depth-first search on the edge list 'edges' to find the
   // connected component containing the start_node_id. Edges are moved from
   // the original edge list as part of the implementation. The edges in the
   // edge list of the subgraph (connected component) containing
   // 'start_node_id' are given the label 'subgraph_label' and appended to
   // 'labeled_edges'.
   template <typename EdgeId_T>
   void connectedComponent(
      DynamicArray<types::edge_t<int, EdgeId_T> > & edges,
      DynamicArray<int> & visited_nodes,
      const int start_node_id,
      const int subgraph_label,
      DynamicArray<types::labeled_edge_t<int, EdgeId_T> > & labeled_edges
   )
   {
      if (edges.size() == 0)
      {
         return;
      }

      DynamicArray<int> frontier(2 * visited_nodes.size());
      frontier.append(start_node_id);

      while (frontier.size() > 0)
      {
         int node_id = frontier.pop_back();

         while (visited_nodes.at(node_id) != 0)
         {
            if (frontier.size() > 0)
            {
               node_id = frontier.pop_back();
            }
            else
            {
               node_id = -1;
               break;
            }
         }

         if (node_id == -1)
         {
            break;
         }

         visited_nodes.at(node_id) = subgraph_label;

         DynamicArray<int> deletable_edge_indices(16);
         int i = 0;
         for (const auto edge : edges)
         {
            if (edge.nodeIdA == node_id)
            {
               deletable_edge_indices.push_back(i);
               frontier.push_back(edge.nodeIdB);
            }
            else if (edge.nodeIdB == node_id)
            {
               deletable_edge_indices.push_back(i);
               frontier.push_back(edge.nodeIdA);
            }
            ++i;
         }

         std::sort(
            deletable_edge_indices.begin(),
            deletable_edge_indices.end(),
            reverseIndex_t()
         );

         for (const auto index : deletable_edge_indices)
         {
            types::edge_t<int, EdgeId_T> temp_edge = edges.pop(index);

            types::labeled_edge_t<int, EdgeId_T> temp_labeled_edge;
            temp_labeled_edge.nodeIdA = temp_edge.nodeIdA;
            temp_labeled_edge.nodeIdB = temp_edge.nodeIdB;
            temp_labeled_edge.edgeId = temp_edge.edgeId;
            temp_labeled_edge.label = subgraph_label;

            labeled_edges.push_back(temp_labeled_edge);
         }
      }
   }

   template <typename Node_T, typename EdgeId_T>
   void calculateSubgraphBookmarks(
      const DynamicArray<types::labeled_edge_t<Node_T, EdgeId_T> > & labeled_edges,
      DynamicArray<types::subgraph_bookmark_t> & subgraph_bookmarks
   )
   {
      subgraph_bookmarks.clear();
      if (labeled_edges.size() == 0)
      {
         return;
      }

      int component_label = labeled_edges[0].label;
      unsigned int start_index = 0;
      for (unsigned int i = 0; i < labeled_edges.size(); ++i)
      {
         const auto labeled_edge = labeled_edges[i];

         if (labeled_edge.label != component_label)
         {
            types::subgraph_bookmark_t bookmark = {
               start_index, i - start_index
            };
            subgraph_bookmarks.push_back(bookmark);

            component_label = labeled_edge.label;
            start_index = i;
         }
      }

      types::subgraph_bookmark_t bookmark = {
         start_index,
         static_cast<unsigned int>(labeled_edges.size() - start_index)
      };
      subgraph_bookmarks.push_back(bookmark);
   }

   template <typename Node_T, typename EdgeId_T>
   void connectedComponentsDecomposition(
      const DynamicArray<types::edge_t<Node_T, EdgeId_T> > & edges,
      DynamicArray<types::labeled_edge_t<Node_T, EdgeId_T> > & labeled_edges
   )
   {
      // The external 'edges' edge list has its nodes relabeled to ints.
      // Connected components are created using int node types.
      DynamicArray<types::edge_t<int, EdgeId_T> > internal_edges(edges.size());
      DynamicArray<Node_T> new_to_old_labels(32);

      relabelNodesSequentially(edges, new_to_old_labels, internal_edges);

      // Nodes that have been mapped to a connected component are considered
      // 'labeled'. A label of '0' means that a node has not been mapped to a
      // connected component.
      DynamicArray<int> visited_nodes(new_to_old_labels.size(), 0);

      int subgraph_label = 1;
      DynamicArray<types::labeled_edge_t<int, EdgeId_T> > labeled_internal_edges(edges.size());
      while (internal_edges.size() > 0)
      {
         int start_node_id = -1;
         for (
            int node_id = 0;
            static_cast<unsigned int>(node_id) < new_to_old_labels.size();
            ++node_id
         )
         {
            if (visited_nodes[node_id] == 0)
            {
               start_node_id = node_id;
               break;
            }
         }

         if (start_node_id == -1)
         {
            std::cout << "Something weird happened - there are edges left in the frontier, but all nodes have been labeled\n";
            break;
         }

         connectedComponent(
            internal_edges,
            visited_nodes,
            start_node_id,
            subgraph_label,
            labeled_internal_edges
         );

         if (static_cast<unsigned int>(subgraph_label) > edges.size() + 1)
         {
            std::cout << "More subgraph labels than internal edges\n";
            break;
         }

         ++subgraph_label;
      }

      for (unsigned int i = 0; i < labeled_internal_edges.size(); ++i)
      {
         graph::types::labeled_edge_t<Node_T, EdgeId_T> temp_edge;
         temp_edge.label = labeled_internal_edges[i].label;
         temp_edge.nodeIdA = new_to_old_labels[labeled_internal_edges[i].nodeIdA];
         temp_edge.nodeIdB = new_to_old_labels[labeled_internal_edges[i].nodeIdB];
         temp_edge.edgeId = labeled_internal_edges[i].edgeId;

         labeled_edges.push_back(temp_edge);
      }
   }

   template <typename Node_T, typename EdgeId_T>
   void connectedComponentsDecomposition(
      const DynamicArray<types::edge_t<Node_T, EdgeId_T> > & edges,
      DynamicArray<types::labeled_edge_t<Node_T, EdgeId_T> > & labeled_edges,
      DynamicArray<types::subgraph_bookmark_t> & subgraph_bookmarks
   )
   {
      connectedComponentsDecomposition(edges, labeled_edges);
      calculateSubgraphBookmarks(labeled_edges, subgraph_bookmarks);
   }
}

#endif
