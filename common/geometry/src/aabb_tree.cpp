#include "aabb_tree.hpp"

#include <iostream>

namespace geometry
{

   aabb_t combine(const aabb_t a, const aabb_t b)
   {
      return {
         {
            std::max(a.vertMax[0], b.vertMax[0]),
            std::max(a.vertMax[1], b.vertMax[1]),
            std::max(a.vertMax[2], b.vertMax[2])
         },
         {
            std::min(a.vertMin[0], b.vertMin[0]),
            std::min(a.vertMin[1], b.vertMin[1]),
            std::min(a.vertMin[2], b.vertMin[2])
         }
      };
   }

   void growTree(const aabb_meta_t new_aabb, AabbTree & tree)
   {
      GrowPriorityExplorer explorer(new_aabb.aabb);

      tree.priorityTraverse(explorer);

      auto & best_node = tree.getNode(explorer.best_node_id);
      int num_best_node_children = numChildren(best_node);

      int node_to_resize_id = -1;

      if (
         (num_best_node_children == 0) &&
         (best_node.meta.uid >= 0)
      )
      {
         aabb_meta_t bounding_aabb;
         bounding_aabb.aabb = combine(new_aabb.aabb, best_node.meta.aabb);
         bounding_aabb.uid = -1;
         int bounding_node_id = tree.emplace(
            explorer.best_node_id, bounding_aabb
         );
         tree.addChild(bounding_node_id, new_aabb);

         node_to_resize_id = tree.getNode(bounding_node_id).parentNodeId;
      }
      else
      {
         tree.addChild(explorer.best_node_id, new_aabb);

         node_to_resize_id = explorer.best_node_id;
      }

      // Resize the ancestors of the new AABB to contain the new AABB
      while (node_to_resize_id >= 0)
      {
         auto & resize_meta_aabb = tree[node_to_resize_id];
         resize_meta_aabb.aabb = geometry::combine(resize_meta_aabb.aabb, new_aabb.aabb);
         node_to_resize_id = tree.getNode(node_to_resize_id).parentNodeId;
      }
   }
}
