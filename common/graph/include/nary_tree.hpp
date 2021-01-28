#ifndef GRAPH_NARY_TREE_HEADER
#define GRAPH_NARY_TREE_HEADER

#include "graph_types.hpp"

#include <functional>
#include <queue>
#include <vector>

namespace graph
{

template <typename Metadata_T, int NumChildren = 2>
class NAryTree
{
   public:
      typedef graph::types::dl_node_t<Metadata_T, NumChildren> MetaNode_T;

      NAryTree(const Metadata_T & meta)
      {
         static_assert(
            NumChildren >= 2, "N-ary tree requires two or more children per node"
         );
         setRoot(meta);
      }

      NAryTree(unsigned int reserve_size, const Metadata_T & meta)
      {
         static_assert(
            NumChildren >= 2, "N-ary tree requires two or more children per node"
         );
         nodes_.reserve(reserve_size);

         MetaNode_T root_node = makeDefaultNode(meta);

         nodes_.push_back(root_node);
      }

      Metadata_T & operator[](unsigned int i)
      {
         return nodes_.at(i).meta;
      }

      const Metadata_T & operator[](unsigned int i) const
      {
         return nodes_.at(i).meta;
      }

      // Clears the existing tree and sets a root node for the tree.
      void setRoot(const Metadata_T & meta)
      {
         nodes_.clear();

         MetaNode_T root_node = makeDefaultNode(meta);

         nodes_.push_back(root_node);
      }

      int getRootId(void) const
      {
         for (int i = 0; i < static_cast<int>(nodes_.size()); ++i)
         {
            if (nodes_[i].parentNodeId == -1)
            {
               return i;
            }
         }

         return -1;
      }

      const MetaNode_T & getRoot(void) const
      {
         for (auto & node : nodes_)
         {
            if (node.parentNodeId == -1)
            {
               return node;
            }
         }

         return nodes_.at(0);
      }

      const MetaNode_T & getNode(unsigned int i) const
      {
         return nodes_.at(i);
      }

      // Returns the ID of the child node if the node is added to the parent
      // successfully. Returns -1 if the child is not added successfully. Note
      // that node IDs only change if nodes are removed from the tree.
      int addChild(const int parent_id, const Metadata_T & meta)
      {
         MetaNode_T new_node = makeDefaultNode(meta);
         new_node.parentNodeId = parent_id;

         int free_child_index = -1;
         {
            // Can't re-use this reference after this if statement. The vector
            // might grow and invalidate the reference.
            MetaNode_T & parent_node = nodes_.at(parent_id);

            bool parent_full = true;
            for (int i = 0; i < NumChildren; ++i)
            {
               if (parent_node.childNodeIds[i] == -1)
               {
                  parent_full = false;
                  free_child_index = i;
                  break;
               }
            }

            if (parent_full)
            {
               return -1;
            }
         }

         nodes_.push_back(new_node);
         const int child_id = static_cast<int>(nodes_.size()) - 1;

         // Note that the reference to the parent node is made after the
         // underlying vector grows. Otherwise reference invalidation might
         // happen.
         MetaNode_T & parent_node = nodes_.at(parent_id);

         parent_node.childNodeIds[free_child_index] = child_id;

         return child_id;
      }

      // Adds metadata to a new node, makes the node at 'emplaced_node_id' a
      // child of the new node, and makes the new node the child of the
      // replaced node's parent.
      // This increases the depth of the tree at the node being emplaced.
      // Returns the ID of the new node if successful, -1 otherwise.
      int emplace(const int emplaced_node_id, const Metadata_T & meta)
      {
         MetaNode_T new_node = makeDefaultNode(meta);
         new_node.parentNodeId = nodes_.at(emplaced_node_id).parentNodeId;
         new_node.childNodeIds[0] = emplaced_node_id;

         nodes_.push_back(new_node);
         int new_node_id = static_cast<int>(nodes_.size()) - 1;
         nodes_.at(emplaced_node_id).parentNodeId = new_node_id;

         if (nodes_.at(new_node_id).parentNodeId < 0)
         {
            return new_node_id;
         }

         MetaNode_T & new_node_parent_node = nodes_.at(nodes_.at(new_node_id).parentNodeId);

         for (int i = 0; i < NumChildren; ++i)
         {
            if (new_node_parent_node.childNodeIds[i] == emplaced_node_id)
            {
               new_node_parent_node.childNodeIds[i] = new_node_id;
            }
         }

         return new_node_id;
      }

      void removeNode(const int removed_node_id)
      {
         // Just take the last node (if there is one) and put it in the spot
         // of 'removed_node_id'.
         const unsigned int last_node_id = nodes_.size() - 1;

         if (static_cast<unsigned int>(removed_node_id) == last_node_id)
         {
            nodes_.pop_back();
            return;
         }

         MetaNode_T & last_node_parent = getParent(last_node_id);

         for (int i = 0; i < NumChildren; ++i)
         {
            if (last_node_parent.childNodeIds[i] == last_node_id)
            {
               last_node_parent.childNodeIds[i] = removed_node_id;
               break;
            }
         }

         nodes_.at(removed_node_id) = nodes_.at(last_node_id);
         nodes_.pop_back();
      }

      // Swaps the lineage of node_id and uncle_node_id if the node at
      // uncle_node_id is avuncular to node_id.
      // E.g. The parents of the node at node_id and the node at uncle_node_id
      // are swapped.
      void avuncularSwap(const int node_id, const int uncle_node_id)
      {
         if (!avuncular(node_id, uncle_node_id))
         {
            return;
         }

         auto & parent_node = nodes_.at(nodes_.at(node_id).parentNodeId);
         auto & grandparent_node = nodes_.at(parent_node.parentNodeId);

         const int parent_node_id = nodes_.at(node_id).parentNodeId;
         const int grandparent_node_id = parent_node.parentNodeId;

         nodes_.at(node_id).parentNodeId = grandparent_node_id;
         nodes_.at(uncle_node_id).parentNodeId = parent_node_id;

         for (int i = 0; i < NumChildren; ++i)
         {
            if (parent_node.childNodeIds[i] == node_id)
            {
               parent_node.childNodeIds[i] = uncle_node_id;
            }
         }

         for (int i = 0; i < NumChildren; ++i)
         {
            if (grandparent_node.childNodeIds[i] == uncle_node_id)
            {
               grandparent_node.childNodeIds[i] = node_id;
            }
         }
      }

      // Performs a depth-first traversal with node exploration dictated by the
      // output of the FrontierExpander object. The FrontierExpander class must
      // be a functor that takes in a const reference to a MetaNode_T type.
      template <typename FrontierExpander>
      void traverse(FrontierExpander & expander) const
      {
         std::vector<int> frontier;
         frontier.reserve(size());

         int parent_node_id = -1;
         for (int i = 0; i < nodes_.size(); ++i)
         {
            if (nodes_[i].parentNodeId == -1)
            {
               parent_node_id = i;
               break;
            }
         }

         frontier.push_back(parent_node_id);

         while (!frontier.empty())
         {
            int node_id = frontier.back();
            frontier.pop_back();
            const MetaNode_T & node = nodes_.at(node_id);

            if (expander(node_id, node))
            {
               for (int i = 0; i < NumChildren; ++i)
               {
                  if (node.childNodeIds[i] >= 0)
                  {
                     frontier.push_back(node.childNodeIds[i]);
                  }
               }
            }
         }
      }

      template <typename CostFrontierExplorer>
      void priorityTraverse(CostFrontierExplorer & explorer) const
      {
         typedef graph::types::priority_node_id_t pNodeId_t;
         // Using 'std::greater' as the comparison operator makes the lowest-
         // cost nodes appear at the top of the frontier.
         std::priority_queue<
            pNodeId_t, std::vector<pNodeId_t>, std::greater<pNodeId_t>
         > frontier;

         const int root_node_id = getRootId();

         const pNodeId_t root_pnode = {
            root_node_id, explorer(nodes_.at(root_node_id))
         };
         frontier.push(root_pnode);

         while (!frontier.empty())
         {
            const pNodeId_t pnode_id = frontier.top();
            frontier.pop();
            const MetaNode_T & node = nodes_.at(pnode_id.node_id);
            const graph::types::priority_expansion_t pexpand = explorer(
               pnode_id, node
            );

            if (pexpand.expand)
            {
               for (int i = 0; i < NumChildren; ++i)
               {
                  if (node.childNodeIds[i] >= 0)
                  {
                     const pNodeId_t child_pnode = {
                        node.childNodeIds[i], pexpand.cost
                     };
                     frontier.push(child_pnode);
                  }
               }
            }
         }
      }

      std::size_t size(void) const
      {
         return nodes_.size();
      }

      unsigned int numChildrenPerNode(void) const
      {
         return NumChildren;
      }

      // Returns true if the node at query_node_id is an uncle of the node at
      // node_id, returns false otherwise.
      bool avuncular(const int node_id, const int query_node_id) const
      {
         return (
            (nodes_.at(node_id).parentNodeId != -1) &&
            (nodes_.at(node_id).parentNodeId != query_node_id) &&
            (getParent(node_id).parentNodeId != -1) &&
            child(getParent(node_id).parentNodeId, query_node_id)
         );
      }

   private:
      std::vector<MetaNode_T> nodes_;

      // Returns true if the node at query_node_id is a child of the node at
      // node_id, returns false otherwise.
      bool child(const int node_id, const int query_node_id) const
      {
         bool child_found = false;

         const MetaNode_T & temp_node = nodes_.at(node_id);
         for (int i = 0; i < NumChildren; ++i)
         {
            child_found |= (temp_node.childNodeIds[i] == query_node_id);
         }

         return child_found;
      }

      // Returns the first unused child node ID from the node. If all children
      // from the node are used, this returns the first child node ID.
      int & firstAvailableChildId(MetaNode_T & node)
      {
         for (int i = 0; i < NumChildren; ++i)
         {
            if (node.childNodeIds[i] < 0)
            {
               return node.childNodeIds[i];
            }
         }

         return node.childNodeIds[0];
      }

      // Returns a reference to the parent node of the node at 'node_id'.
      MetaNode_T & getParent(const int node_id)
      {
         return nodes_.at(
            nodes_.at(node_id).parentNodeId
         );
      }

      // Returns a const reference to the parent node of the node at 'node_id'.
      const MetaNode_T & getParent(const int node_id) const
      {
         return nodes_.at(
            nodes_.at(node_id).parentNodeId
         );
      }

      inline MetaNode_T makeDefaultNode(const Metadata_T & meta) const
      {
         MetaNode_T temp_node;
         for (int i = 0; i < NumChildren; ++i)
         {
            temp_node.childNodeIds[i] = -1;
         }
         temp_node.parentNodeId = -1;
         temp_node.meta = meta;

         return temp_node;
      }

};

}

#endif
