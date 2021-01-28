#ifndef AABB_TREE_HEADER
#define AABB_TREE_HEADER

#include "geometry_types.hpp"
#include "nary_tree.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace geometry
{
   typedef geometry::types::aabb_t aabb_t;

   struct aabb_meta_t
   {
      int64_t uid;
      aabb_t aabb;
   };

   typedef graph::NAryTree<aabb_meta_t, 4> AabbTree;

   inline float surfaceArea(const aabb_t box)
   {
      const Vector3 s = (box.vertMax - box.vertMin);
      return 2.f * ((s[0] * s[1]) + (s[0] * s[2]) + (s[1] * s[2]));
   }

   inline int numChildren(const AabbTree::MetaNode_T & node)
   {
      return (
         (node.childNodeIds[0] >= 0) +
         (node.childNodeIds[1] >= 0) +
         (node.childNodeIds[2] >= 0) +
         (node.childNodeIds[3] >= 0)
      );
   }

   aabb_t combine(const aabb_t a, const aabb_t b);

   void growTree(const aabb_meta_t new_aabb, AabbTree & tree);

   struct GrowPriorityExplorer
   {
      GrowPriorityExplorer(const aabb_t query)
         : best_node_id(-1)
         , best_cost_(__FLT_MAX__)
         , query_cost_(surfaceArea(query))
         , query_(query)
      { }

      GrowPriorityExplorer(
         const GrowPriorityExplorer &
      ) = delete;

      GrowPriorityExplorer & operator=(
         const GrowPriorityExplorer &
      ) = delete;

      // Evaluation of the root node's lower-bound cost
      float operator()(const AabbTree::MetaNode_T & node) const
      {
         aabb_t union_aabb = combine(query_, node.meta.aabb);
         const float initial_cost = surfaceArea(union_aabb) + query_cost_;
         return initial_cost;
      }

      graph::types::priority_expansion_t operator()(
         const graph::types::priority_node_id_t pnode_id,
         const AabbTree::MetaNode_T & node
      )
      {
         aabb_t union_aabb = combine(query_, node.meta.aabb);

         const float union_aabb_cost = surfaceArea(union_aabb);

         // Determine the cost of making the node a partner of the query AABB
         // const float partner_cost = pnode_id.cost + union_aabb_cost;
         const float partner_cost = pnode_id.cost + union_aabb_cost - query_cost_;

         // Determine the lower-bound cost of making the query AABB a child of
         // the current node
         const float lower_bound_cost = pnode_id.cost + union_aabb_cost - surfaceArea(node.meta.aabb);

         if (partner_cost < best_cost_)
         {
            best_cost_ = partner_cost;
            best_node_id = pnode_id.node_id;
         }

         graph::types::priority_expansion_t expansion_result;
         expansion_result.expand = (
            (lower_bound_cost < best_cost_) &&
            (node.meta.uid < 0) &&
            (numChildren(node) == 4)
         );

         expansion_result.cost = lower_bound_cost;

         return expansion_result;
      }

      // The ID of the node that generates the lowest cost to update the AABB
      // tree to include the query AABB. This is guaranteed to be either a leaf
      // node or a branch node with less than four children.
      int best_node_id;

      private:
         float best_cost_;

         // The cost of the query AABB - the same as the AABB's surface area.
         const float query_cost_;

         aabb_t query_;
   };
}

#endif
