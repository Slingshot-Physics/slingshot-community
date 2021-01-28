#ifndef GRAPH_TYPES_HEADER
#define GRAPH_TYPES_HEADER

#include <cstdint>

namespace graph
{

namespace types
{
   enum class enumTransitivity_t
   {
      TRANSITIVE = 0,
      NODE_A_TERMINAL = 1,
      NODE_B_TERMINAL = 2,
   };

   template <typename Node_T = int, typename EdgeId_T = int64_t>
   struct edge_t
   {
      Node_T nodeIdA;
      Node_T nodeIdB;
      EdgeId_T edgeId;
      enumTransitivity_t edgeType;
   };

   template <typename Node_T = int, typename EdgeId_T = int64_t>
   struct labeled_edge_t
   {
      Node_T nodeIdA;
      Node_T nodeIdB;
      EdgeId_T edgeId;
      int label;
   };

   struct subgraph_bookmark_t
   {
      unsigned int startIndex;
      unsigned int size;
   };

   // Returned by a functor in n-ary tree expansion.
   struct priority_expansion_t
   {
      bool expand;
      float cost;
   };

   // Defines elements of the frontier in priority traversal of an n-ary tree.
   struct priority_node_id_t
   {
      int node_id;
      float cost;

      inline bool operator>(const priority_node_id_t other) const
      {
         return cost > other.cost;
      }

      inline bool operator<(const priority_node_id_t other) const
      {
         return cost < other.cost;
      }
   };

   // A doubly-linked node type intended for use in an array.
   template <typename Metadata_T, unsigned int NumChildren = 2>
   struct dl_node_t
   {
      int parentNodeId;

      int childNodeIds[NumChildren];

      Metadata_T meta;

      unsigned int maxNumChildren(void) const
      {
         return NumChildren;
      }
   };

}

}

#endif
