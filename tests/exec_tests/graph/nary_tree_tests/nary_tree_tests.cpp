#include "nary_tree.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <iostream>

template <typename Metadata_T, unsigned int NumChildren>
struct frontierExpander_t
{
   int num_calls;

   std::vector<Metadata_T> node_ids;

   frontierExpander_t(void)
      : num_calls(0)
   { }

   bool operator()(
      const int node_id,
      const typename graph::types::dl_node_t<Metadata_T, NumChildren> & node
   )
   {
      (void)node;
      node_ids.push_back(node_id);
      ++num_calls;
      return true;
   }
};

template <typename Metadata_T, unsigned int NumChildren>
struct priorityFrontierExpander_t
{
   int num_calls;

   std::vector<Metadata_T> node_ids;

   priorityFrontierExpander_t(void)
      : num_calls(0)
   { }

   float operator()(
      const typename graph::types::dl_node_t<Metadata_T, NumChildren> & node
   ) const
   {
      (void)node;
      return 0.f;
   }

   // Explores the entire tree and returns the node's metadata value as its
   // cost.
   graph::types::priority_expansion_t operator()(
      const graph::types::priority_node_id_t pnode_id,
      const typename graph::types::dl_node_t<Metadata_T, NumChildren> & node
   )
   {
      (void)node;
      node_ids.push_back(pnode_id.node_id);
      ++num_calls;
      return {true, static_cast<float>(node.meta)};
   }
};

TEST_CASE( "instantiate binary tree", "[binary_tree]" )
{
   graph::NAryTree<int> tree(-2);

   REQUIRE( tree.numChildrenPerNode() == 2 );
}

TEST_CASE( "add binary tree root node", "[binary_tree]")
{
   graph::NAryTree<int> tree(-7);

   REQUIRE( tree.size() == 1 );
   REQUIRE( tree[0] == -7);

   REQUIRE( tree.getNode(0).meta == -7 );
   for (int i = 0; i < tree.numChildrenPerNode(); ++i)
   {
      REQUIRE( tree.getNode(0).childNodeIds[i] == -1 );
   }
}

TEST_CASE( "binary tree add more than the maximum number of child nodes", "[binary_tree]" )
{
   std::vector<int> child_ids;
   graph::NAryTree<int, 2> tree(5);

   for (int i = 0; i < tree.numChildrenPerNode() + 1; ++i)
   {
      child_ids.push_back(tree.addChild(0, i));
   }

   for (int i = 0; i < child_ids.size(); ++i)
   {
      if (i < tree.numChildrenPerNode())
      {
         REQUIRE( child_ids[i] >= 0 );
      }
      else
      {
         REQUIRE( child_ids[i] == -1 );
      }
   }

   REQUIRE( tree.size() == 3 );
}

TEST_CASE( "reset binary tree root node", "[binary_tree]")
{
   graph::NAryTree<int> tree(-7);

   REQUIRE( tree.size() == 1 );
   REQUIRE( tree[0] == -7);

   tree.setRoot(-8);

   REQUIRE( tree.size() == 1 );
   REQUIRE( tree[0] == -8 );
}

TEST_CASE( "add two child nodes to binary tree", "[binary_tree]")
{
   graph::NAryTree<int> tree(-7);

   int new_child_id_a = tree.addChild(0, -4);
   int new_child_id_b = tree.addChild(0, 3);

   std::cout << "child ids: " << new_child_id_a << ", " << new_child_id_b << "\n";

   REQUIRE( new_child_id_a != -1 );
   REQUIRE( new_child_id_b != -1 );
   REQUIRE( tree.getNode(0).childNodeIds[0] == new_child_id_a );
   REQUIRE( tree.getNode(0).childNodeIds[1] == new_child_id_b );
   REQUIRE( tree.getNode(new_child_id_a).parentNodeId == 0 );
   REQUIRE( tree.getNode(new_child_id_b).parentNodeId == 0 );
   REQUIRE( tree[new_child_id_a] == -4 );
   REQUIRE( tree[new_child_id_b] == 3 );
}

// Verifies that the root node of a binary tree can be emplaced by another
// node, and that the root node is updated to the new node.
TEST_CASE( "emplace binary tree's root node", "[binary_tree]" )
{
   graph::NAryTree<int, 2> tree(-5);

   int new_node_id = tree.emplace(0, 2);

   REQUIRE( tree.size() == 2 );
   REQUIRE( tree.getNode(new_node_id).parentNodeId == -1 );
   REQUIRE( tree.getNode(new_node_id).childNodeIds[0] == 0 );
   REQUIRE( tree.getNode(0).parentNodeId == new_node_id );

   REQUIRE( tree.getRoot().parentNodeId == -1 );
   REQUIRE( tree.getRoot().childNodeIds[0] == 0 );
}

TEST_CASE( "emplace a node in a binary tree", "[binary_tree]" )
{
   graph::NAryTree<int, 2> tree(-5);

   int first_child_id = tree.addChild(0, 4);

   REQUIRE( tree.getNode(first_child_id).parentNodeId == 0 );

   REQUIRE( tree.size() == 2 );
   REQUIRE( tree.getNode(0).childNodeIds[0] == 1 );
   REQUIRE( tree.getNode(1).parentNodeId == 0 );

   int new_node_id = tree.emplace(1, 7);

   REQUIRE( tree.getNode(1).parentNodeId == new_node_id );
   REQUIRE( tree.getNode(0).childNodeIds[0] == new_node_id );
   REQUIRE( tree.getNode(2).parentNodeId == 0 );
}

TEST_CASE( "emplace a node in a branchy binary tree", "[binary_tree]" )
{
   //         ___<0,-7>___
   //         |          |
   //     <1,-4>     ___<2,3>
   //                |
   //          ___<3,18>
   //          |
   //        <4,9>
   graph::NAryTree<int> tree(-7);

   int child_ids[4];

   child_ids[0] = tree.addChild(0, -4);
   child_ids[1] = tree.addChild(0, 3);

   child_ids[2] = tree.addChild(child_ids[1], 18);
   child_ids[3] = tree.addChild(child_ids[2], 9);

   int new_node_id = tree.emplace(2, 22);

   REQUIRE( tree[new_node_id] == 22 );
   REQUIRE( tree.getNode(0).childNodeIds[1] == new_node_id );
   REQUIRE( tree.getNode(new_node_id).parentNodeId == 0 );
   REQUIRE( tree.getNode(new_node_id).childNodeIds[0] == 2 );
   REQUIRE( tree.getNode(new_node_id).childNodeIds[1] == -1 );
   REQUIRE( tree.getNode(2).parentNodeId == new_node_id );
}

TEST_CASE( "add 4 child nodes to binary tree", "[binary_tree]")
{
   //         ___<0,-7>___
   //         |          |
   //     <1,-4>     ___<2,3>
   //                |
   //          ___<3,18>
   //          |
   //        <4,9>
   graph::NAryTree<int> tree(-7);

   int child_ids[4];

   child_ids[0] = tree.addChild(0, -4);
   std::cout << "added first child\n";
   child_ids[1] = tree.addChild(0, 3);
   std::cout << "added second child " << child_ids[1] << "\n";

   child_ids[2] = tree.addChild(child_ids[1], 18);
   std::cout << "added third child\n";
   child_ids[3] = tree.addChild(child_ids[2], 9);
   std::cout << "added fourth child\n";

   std::cout << "added all children\n";

   for (int i = 0; i < 4; ++i)
   {
      REQUIRE( child_ids[i] != -1 );
   }
   REQUIRE( tree[child_ids[0]] == -4 );
   REQUIRE( tree[child_ids[1]] == 3 );
   REQUIRE( tree[child_ids[2]] == 18 );
   REQUIRE( tree[child_ids[3]] == 9 );
   REQUIRE( tree.getNode(0).childNodeIds[0] == child_ids[0] );
   REQUIRE( tree.getNode(0).childNodeIds[1] == child_ids[1] );
   REQUIRE( tree.getNode(child_ids[0]).childNodeIds[0] == -1 );
   REQUIRE( tree.getNode(child_ids[0]).childNodeIds[1] == -1 );
   REQUIRE( tree.getNode(child_ids[1]).childNodeIds[0] == child_ids[2] );
   REQUIRE( tree.getNode(child_ids[1]).childNodeIds[1] == -1 );
   REQUIRE( tree.getNode(child_ids[2]).childNodeIds[0] == child_ids[3] );
   REQUIRE( tree.getNode(child_ids[2]).childNodeIds[1] == -1 );
}

TEST_CASE( "valid avuncular swap on binary tree", "[binary_tree]" )
{
   graph::NAryTree<int> tree(-7);
   std::vector<int> child_ids;
   child_ids.push_back(tree.addChild(0, 2));
   child_ids.push_back(tree.addChild(0, 1));
   child_ids.push_back(tree.addChild(child_ids[0], 13));
   child_ids.push_back(tree.addChild(child_ids[0], 17));
   child_ids.push_back(tree.addChild(child_ids[1], -4));
   child_ids.push_back(tree.addChild(child_ids[1], 6));
   child_ids.push_back(tree.addChild(child_ids[2], 9));
   child_ids.push_back(tree.addChild(child_ids[2], -3));

   int node_id = child_ids[2];
   int uncle_node_id = child_ids[1];

   int old_parent_id = tree.getNode(node_id).parentNodeId;
   int old_uncle_parent_id = tree.getNode(uncle_node_id).parentNodeId;

   REQUIRE( tree.avuncular(node_id, uncle_node_id) );

   tree.avuncularSwap(node_id, uncle_node_id);

   REQUIRE( tree.getNode(node_id).parentNodeId != old_parent_id );
   REQUIRE( tree.getNode(uncle_node_id).parentNodeId != old_uncle_parent_id );

   REQUIRE( tree.getNode(node_id).parentNodeId == old_uncle_parent_id );
   REQUIRE( tree.getNode(uncle_node_id).parentNodeId == old_parent_id );
}

TEST_CASE( "invalid avuncular swap on binary tree", "[binary_tree]" )
{
   graph::NAryTree<int> tree(-7);
   std::vector<int> child_ids;
   child_ids.push_back(tree.addChild(0, 2));
   child_ids.push_back(tree.addChild(0, 1));
   child_ids.push_back(tree.addChild(child_ids[0], 13));
   child_ids.push_back(tree.addChild(child_ids[0], 17));
   child_ids.push_back(tree.addChild(child_ids[1], -4));
   child_ids.push_back(tree.addChild(child_ids[1], 6));
   child_ids.push_back(tree.addChild(child_ids[2], 9));
   child_ids.push_back(tree.addChild(child_ids[2], -3));

   int node_id = child_ids[0];
   int uncle_node_id = child_ids[1];

   int old_parent_id = tree.getNode(node_id).parentNodeId;
   int old_uncle_parent_id = tree.getNode(uncle_node_id).parentNodeId;

   REQUIRE( !tree.avuncular(node_id, uncle_node_id) );

   tree.avuncularSwap(node_id, uncle_node_id);

   REQUIRE( tree.getNode(node_id).parentNodeId == old_parent_id );
   REQUIRE( tree.getNode(uncle_node_id).parentNodeId == old_uncle_parent_id );
}

TEST_CASE( "invalid avuncular swap (child and parent) on binary tree", "[binary_tree]" )
{
   graph::NAryTree<int> tree(-7);
   std::vector<int> child_ids;
   child_ids.push_back(tree.addChild(0, 2));
   child_ids.push_back(tree.addChild(0, 1));
   child_ids.push_back(tree.addChild(child_ids[0], 13));
   child_ids.push_back(tree.addChild(child_ids[0], 17));
   child_ids.push_back(tree.addChild(child_ids[1], -4));
   child_ids.push_back(tree.addChild(child_ids[1], 6));
   child_ids.push_back(tree.addChild(child_ids[2], 9));
   child_ids.push_back(tree.addChild(child_ids[2], -3));

   int node_id = child_ids[0];
   int uncle_node_id = child_ids[3];

   int old_parent_id = tree.getNode(node_id).parentNodeId;
   int old_uncle_parent_id = tree.getNode(uncle_node_id).parentNodeId;

   REQUIRE( !tree.avuncular(node_id, uncle_node_id) );

   tree.avuncularSwap(node_id, uncle_node_id);

   REQUIRE( tree.getNode(node_id).parentNodeId == old_parent_id );
   REQUIRE( tree.getNode(uncle_node_id).parentNodeId == old_uncle_parent_id );
}

TEST_CASE( "traverse binary tree", "[binary_tree]" )
{
   // Nodes are indicated by: <node_id, metadata>
   //
   //         ___<0,-7>___
   //         |          |
   //     <1,-4>     ___<2,3>
   //                |
   //          ___<3,18>
   //          |
   //        <4,9>
   graph::NAryTree<int> tree(-7);

   int child_ids[4];

   child_ids[0] = tree.addChild(0, -4);
   child_ids[1] = tree.addChild(0, 3);
   child_ids[2] = tree.addChild(child_ids[1], 18);
   child_ids[3] = tree.addChild(child_ids[2], 9);

   frontierExpander_t<int, 2> expander;

   tree.traverse(expander);

   REQUIRE( expander.num_calls == tree.size() );
   for (const auto explored_node_id : expander.node_ids)
   {
      REQUIRE( explored_node_id >= 0 );
   }

   REQUIRE( expander.node_ids[0] == 0 );
   REQUIRE( expander.node_ids[1] == 2 );
   REQUIRE( expander.node_ids[2] == 3 );
   REQUIRE( expander.node_ids[3] == 4 );
   REQUIRE( expander.node_ids[4] == 1 );
}

TEST_CASE( "priority traverse binary tree", "[binary_tree]" )
{
   // Nodes are indicated by: <node_id, metadata>
   //
   //         ___<0,-7>___
   //         |          |
   //     <1,-4>     ___<2,3>
   //                |
   //          ___<3,18>
   //          |
   //        <4,9>
   graph::NAryTree<int> tree(-7);

   int child_ids[4];

   child_ids[0] = tree.addChild(0, -4);
   child_ids[1] = tree.addChild(0, 3);
   child_ids[2] = tree.addChild(child_ids[1], 18);
   child_ids[3] = tree.addChild(child_ids[2], 9);

   priorityFrontierExpander_t<int, 2> expander;

   tree.priorityTraverse(expander);

   REQUIRE( expander.num_calls == tree.size() );
   for (const auto explored_node_id : expander.node_ids)
   {
      REQUIRE( explored_node_id >= 0 );
   }

   REQUIRE( expander.node_ids[0] == 0 );
   REQUIRE( expander.node_ids[1] == 1 );
   REQUIRE( expander.node_ids[2] == 2 );
   REQUIRE( expander.node_ids[3] == 3 );
   REQUIRE( expander.node_ids[4] == 4 );
}

TEST_CASE( "instantiate 4-nary tree", "[4-nary_tree]" )
{
   graph::NAryTree<int, 4> tree(-2);

   REQUIRE( tree.numChildrenPerNode() == 4 );

   for (int i = 0; i < tree.numChildrenPerNode(); ++i)
   {
      REQUIRE( tree.getRoot().childNodeIds[i] == -1 );
   }
}

// Verifies that the root node of a 4-nary tree can be emplaced by another
// node, and that the root node is updated to the new node.
TEST_CASE( "emplace a 4-nary tree's root node", "[4-nary_tree]" )
{
   graph::NAryTree<int, 4> tree(-2);

   int new_node_id = tree.emplace(0, 17);

   REQUIRE( tree.size() == 2 );
   REQUIRE( tree.getNode(new_node_id).meta == 17 );
   REQUIRE( tree.getNode(new_node_id).parentNodeId == -1 );
   REQUIRE( tree.getNode(new_node_id).childNodeIds[0] == 0 );
   REQUIRE( tree.getNode(0).parentNodeId == new_node_id );

   REQUIRE( tree.getRoot().parentNodeId == -1 );
   REQUIRE( tree.getRoot().childNodeIds[0] == 0 );
}

TEST_CASE( "emplace a node in a branchy 4-nary tree", "[4-nary-tree]" )
{
   // Notation for this tree: <node_id, node_val>
   //
   //                       <0,787>
   //             _____________|______________
   //             |        |        |        |
   //           <1,0>    <2,1>    <3,2>    <4,3>
   //   __________|_________________
   //   |         |        |       |
   // <5,5>    <6,10>   <7,15>   <8,20>

   typedef int node_t;
   std::vector<int> child_ids;

   graph::NAryTree<node_t, 4> tree(787);

   for (int i = 0; i < 4; ++i)
   {
      child_ids.push_back(tree.addChild(0, i));
   }

   for (int i = 0; i < 4; ++i)
   {
      child_ids.push_back(tree.addChild(child_ids[0], 5 * (i + 1)));
   }

   // Emplace at a branch node
   int emplaced_node_id = 3;
   int new_node_id = tree.emplace(emplaced_node_id, 22);

   REQUIRE( tree[new_node_id] == 22 );
   REQUIRE( tree.getNode(0).childNodeIds[2] == new_node_id );
   REQUIRE( tree.getNode(new_node_id).parentNodeId == 0 );
   REQUIRE( tree.getNode(new_node_id).childNodeIds[0] == emplaced_node_id );
   REQUIRE( tree.getNode(emplaced_node_id).parentNodeId == new_node_id );
   REQUIRE( tree.getNode(emplaced_node_id).childNodeIds[0] == -1 );

   // Emplace at a leaf node
   emplaced_node_id = 8;
   new_node_id = tree.emplace(emplaced_node_id, 55);

   REQUIRE( tree[new_node_id] == 55 );
   REQUIRE( tree.getNode(1).childNodeIds[3] == new_node_id );
   REQUIRE( tree.getNode(new_node_id).parentNodeId == 1 );
   REQUIRE( tree.getNode(new_node_id).childNodeIds[0] == emplaced_node_id );
   REQUIRE( tree.getNode(emplaced_node_id).parentNodeId == new_node_id );
   REQUIRE( tree.getNode(emplaced_node_id).childNodeIds[0] == -1 );
}

TEST_CASE( "add four child nodes to root in 4-nary tree", "[4-nary_tree]" )
{
   int child_ids[4];
   int child_values[4] = {1, 2, 3, 4};
   graph::NAryTree<int, 4> tree(-2);
   child_ids[0] = tree.addChild(0, child_values[0]);
   child_ids[1] = tree.addChild(0, child_values[1]);
   child_ids[2] = tree.addChild(0, child_values[2]);
   child_ids[3] = tree.addChild(0, child_values[3]);

   for (int i = 0; i < 4; ++i)
   {
      REQUIRE( child_ids[i] >= 0 );
      REQUIRE( tree[child_ids[i]] == child_values[i] );
      REQUIRE( tree.getNode(child_ids[i]).parentNodeId == 0 );
   }
}

TEST_CASE( "4-nary tree add more than the maximum number of child nodes", "[4-nary_tree]" )
{
   std::vector<int> child_ids;
   graph::NAryTree<int, 4> tree(5);

   for (int i = 0; i < tree.numChildrenPerNode() + 1; ++i)
   {
      child_ids.push_back(tree.addChild(0, i));
   }

   for (int i = 0; i < child_ids.size(); ++i)
   {
      if (i < tree.numChildrenPerNode())
      {
         REQUIRE( child_ids[i] >= 0 );
      }
      else
      {
         REQUIRE( child_ids[i] == -1 );
      }
   }

   REQUIRE( tree.size() == 5 );
}

TEST_CASE( "add multiple branches and leaves to 4-nary tree", "[4-nary_tree]" )
{
   typedef int node_t;
   std::vector<int> child_ids;
   std::vector<node_t> child_values;

   graph::NAryTree<node_t, 4> tree(787);

   REQUIRE( tree.numChildrenPerNode() == 4 );

   const int root_node_id = 0;

   for (int i = 0; i < tree.numChildrenPerNode(); ++i)
   {
      child_values.push_back(2 * child_ids.size() + 1);
      child_ids.push_back(
         tree.addChild(root_node_id, child_values.back())
      );

      std::cout << "new child id: " << child_ids.back() << ", index " << i << "\n";
   }

   for (int i = 0; i < 2; ++i)
   {
      child_values.push_back(3 * child_ids.size() + 3);
      child_ids.push_back(
         tree.addChild(child_ids[0], child_values.back())
      );
   }

   int last_child_id = child_ids.back();
   for (int i = 0; i < 3; ++i)
   {
      child_values.push_back(3 * child_ids.size() + 3);
      child_ids.push_back(
         tree.addChild(last_child_id, child_values.back())
      );
   }

   REQUIRE( child_values.size() == child_ids.size() );

   for (int i = 0; i < child_ids.size(); ++i)
   {
      int child_id = child_ids[i];
      std::cout << "child id: " << child_id << "\n";
      REQUIRE( child_id >= 0 );
      REQUIRE( child_values[i] == tree[child_id] );
   }
}

TEST_CASE( "avuncular swap with 4-nary tree", "[4-nary-tree]" )
{
   // Notation for this tree: <node_id, node_val>
   //
   //                       <0,787>
   //             _____________|______________
   //             |        |        |        |
   //           <1,0>    <2,1>    <3,2>    <4,3>
   //   __________|_________________
   //   |         |        |       |
   // <5,5>    <6,10>   <7,15>   <8,20>

   typedef int node_t;
   std::vector<int> child_ids;

   graph::NAryTree<node_t, 4> tree(787);

   for (int i = 0; i < 4; ++i)
   {
      child_ids.push_back(tree.addChild(0, i));
   }

   for (int i = 0; i < 4; ++i)
   {
      child_ids.push_back(tree.addChild(child_ids[0], 5 * (i + 1)));
   }

   const int node_id = child_ids[5];
   const int uncle_node_id = child_ids[3];

   const int old_parent_id = tree.getNode(node_id).parentNodeId;
   const int old_uncle_parent_id = tree.getNode(uncle_node_id).parentNodeId;

   REQUIRE( tree.avuncular(node_id, uncle_node_id) );

   tree.avuncularSwap(node_id, uncle_node_id);

   REQUIRE( tree.getNode(node_id).parentNodeId == old_uncle_parent_id );
   REQUIRE( tree.getNode(uncle_node_id).parentNodeId == old_parent_id );
}

TEST_CASE( "invalid avuncular swap with 4-nary tree", "[4-nary-tree]" )
{
   // Notation for this tree: <node_id, node_val>
   //
   //                       <0,787>
   //             _____________|______________
   //             |        |        |        |
   //           <1,0>    <2,1>    <3,2>    <4,3>
   //   __________|_________________
   //   |         |        |       |
   // <5,5>    <6,10>   <7,15>   <8,20>

   typedef int node_t;
   std::vector<int> child_ids;

   graph::NAryTree<node_t, 4> tree(787);

   for (int i = 0; i < 4; ++i)
   {
      child_ids.push_back(tree.addChild(0, i));
   }

   for (int i = 0; i < 4; ++i)
   {
      child_ids.push_back(tree.addChild(child_ids[0], 5 * (i + 1)));
   }

   const int node_id = child_ids[5];
   const int uncle_node_id = child_ids[0];

   const int old_parent_id = tree.getNode(node_id).parentNodeId;
   const int old_uncle_parent_id = tree.getNode(uncle_node_id).parentNodeId;

   REQUIRE( !tree.avuncular(node_id, uncle_node_id) );

   tree.avuncularSwap(node_id, uncle_node_id);

   REQUIRE( tree.getNode(node_id).parentNodeId == old_parent_id );
   REQUIRE( tree.getNode(uncle_node_id).parentNodeId == old_uncle_parent_id );
}

TEST_CASE( "traverse 4-nary tree", "[4-nary_tree]" )
{
   typedef int node_t;
   std::vector<int> child_ids;
   std::vector<node_t> child_values;

   graph::NAryTree<node_t, 4> tree(787);

   const int root_node_id = 0;

   for (int i = 0; i < tree.numChildrenPerNode(); ++i)
   {
      child_values.push_back(2 * child_ids.size() + 1);
      child_ids.push_back(
         tree.addChild(root_node_id, child_values.back())
      );

      std::cout << "new child id: " << child_ids.back() << ", index " << i << "\n";
   }

   for (int i = 0; i < 2; ++i)
   {
      child_values.push_back(3 * child_ids.size() + 3);
      child_ids.push_back(
         tree.addChild(child_ids[0], child_values.back())
      );
   }

   int last_child_id = child_ids.back();
   for (int i = 0; i < 3; ++i)
   {
      child_values.push_back(3 * child_ids.size() + 3);
      child_ids.push_back(
         tree.addChild(last_child_id, child_values.back())
      );
   }

   frontierExpander_t<node_t, 4> expander;

   tree.traverse(expander);

   REQUIRE( expander.num_calls == tree.size() );
   for (const auto explored_node_id : expander.node_ids)
   {
      REQUIRE( explored_node_id >= 0 );
   }

   for (int i = 0; i < expander.node_ids.size(); ++i)
   {
      for (int j = i + 1; j < expander.node_ids.size(); ++j)
      {
         REQUIRE( expander.node_ids[i] != expander.node_ids[j] );
      }
   }
}

TEST_CASE( "priority traverse 4-nary tree", "[4-nary_tree]" )
{
   typedef int node_t;
   std::vector<int> child_ids;
   std::vector<node_t> child_values;

   graph::NAryTree<node_t, 4> tree(787);

   const int root_node_id = 0;

   for (int i = 0; i < tree.numChildrenPerNode(); ++i)
   {
      child_values.push_back(2 * child_ids.size() + 1);
      child_ids.push_back(
         tree.addChild(root_node_id, child_values.back())
      );

      std::cout << "new child id: " << child_ids.back() << ", index " << i << "\n";
   }

   for (int i = 0; i < 2; ++i)
   {
      child_values.push_back(3 * child_ids.size() + 3);
      child_ids.push_back(
         tree.addChild(child_ids[0], child_values.back())
      );
   }

   int last_child_id = child_ids.back();
   for (int i = 0; i < 3; ++i)
   {
      child_values.push_back(3 * child_ids.size() + 3);
      child_ids.push_back(
         tree.addChild(last_child_id, child_values.back())
      );
   }

   priorityFrontierExpander_t<node_t, 4> expander;

   tree.priorityTraverse(expander);

   REQUIRE( expander.num_calls == tree.size() );
   for (const auto explored_node_id : expander.node_ids)
   {
      REQUIRE( explored_node_id >= 0 );
   }

   for (int i = 0; i < expander.node_ids.size(); ++i)
   {
      for (int j = i + 1; j < expander.node_ids.size(); ++j)
      {
         REQUIRE( expander.node_ids[i] != expander.node_ids[j] );
      }
   }
}

TEST_CASE( "assign doubly linked node", "[dl_node_t]" )
{
   graph::types::dl_node_t<int> node_a = {-1, {-1, -1}, 5};

   graph::types::dl_node_t<int> node_b;

   node_b = node_a;

   REQUIRE( node_a.parentNodeId == node_b.parentNodeId );
   REQUIRE( node_a.childNodeIds[0] == node_b.childNodeIds[0] );
   REQUIRE( node_a.childNodeIds[1] == node_b.childNodeIds[1] );
   REQUIRE( node_a.meta == node_b.meta );
}
