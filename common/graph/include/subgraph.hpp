#ifndef SUBGRAPH_HEADER
#define SUBGRAPH_HEADER

#include "dynamic_array.hpp"
#include "graph_types.hpp"

#include <stdexcept>

namespace graph
{

template <typename Node_T, typename EdgeId_T>
class Subgraph
{
   typedef graph::types::labeled_edge_t<Node_T, EdgeId_T> LabeledEdge_T;

   public:
      Subgraph(void) = delete;

      Subgraph(const Subgraph &) = delete;

      typedef LabeledEdge_T * iterator;

      typedef const LabeledEdge_T * const_iterator;

      Subgraph(
         DynamicArray<LabeledEdge_T> & graph,
         const graph::types::subgraph_bookmark_t & bookmark
      )
         : graph_(graph)
         , bookmark_(bookmark)
      { }

      // Returns a reference to the i-th element of the subgraph
      LabeledEdge_T & operator[](unsigned int i)
      {
         return graph_[indexOffset(i)];
      }

      // Returns a const reference to the i-th element of the subgraph
      const LabeledEdge_T & operator[](unsigned int i) const
      {
         return graph_[indexOffset(i)];
      }

      iterator begin(void)
      {
         return &graph_[indexOffset(0)];
      }

      const_iterator begin(void) const
      {
         return &graph_[indexOffset(0)];
      }

      iterator end(void)
      {
         return &graph_[indexOffset(0)] + bookmark_.size;
      }

      const_iterator end(void) const
      {
         return &graph_[indexOffset(0)] + bookmark_.size;
      }

      // Returns a reference to the i-th element of the subgraph
      LabeledEdge_T & at(unsigned int i)
      {
         if (i >= bookmark_.size)
         {
            throw std::invalid_argument("Index out of bounds");
         }

         return graph_.at(indexOffset(i));
      }

      // Returns a const reference to the i-th element of the subgraph
      const LabeledEdge_T & at(unsigned int i) const
      {
         if (i >= bookmark_.size)
         {
            throw std::invalid_argument("Index out of bounds");
         }

         return graph_.at(indexOffset(i));
      }

      std::size_t size(void) const
      {
         return bookmark_.size;
      }

   private:
      DynamicArray<LabeledEdge_T> & graph_;

      const graph::types::subgraph_bookmark_t & bookmark_;

      // Converts the 0-based index into an offset index for accessing
      // elements from the original graph.
      inline unsigned int indexOffset(unsigned int i) const
      {
         return i + bookmark_.startIndex;
      }
};

}

#endif
