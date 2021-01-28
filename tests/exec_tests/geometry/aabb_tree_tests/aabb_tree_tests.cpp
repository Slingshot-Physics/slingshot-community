#include "aabb_tree.hpp"
#include "geometry_types.hpp"
#include "sat.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

TEST_CASE( "instantiate", "[aabb_tree]" )
{
   geometry::AabbTree tree({1, {{1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f}}});

   REQUIRE( true );
}

TEST_CASE( "add one aabb", "[aabb_tree]" )
{
   geometry::AabbTree tree(
      {1, {{1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f}}}
   );

   geometry::aabb_meta_t new_aabb = {0, {{0.5f, 0.5f, 0.5f}, {-0.75f, -0.75f, 0.f}}};

   geometry::growTree(new_aabb, tree);

   REQUIRE( tree.size() == 3 );
   REQUIRE( tree.getRoot().childNodeIds[0] >= 0 );
   REQUIRE( tree.getRoot().childNodeIds[1] >= 0 );

   REQUIRE( tree.getNode(tree.getRoot().childNodeIds[0]).meta.uid >= 0 );
   REQUIRE( tree.getNode(tree.getRoot().childNodeIds[1]).meta.uid >= 0 );
   REQUIRE( tree.getNode(tree.getRoot().childNodeIds[1]).meta.uid != tree.getNode(tree.getRoot().childNodeIds[0]).meta.uid );
   REQUIRE(
      geometry::collisions::aabbAabb(
         tree.getRoot().meta.aabb,
         tree.getNode(tree.getRoot().childNodeIds[0]).meta.aabb
      )
   );   
   REQUIRE(
      geometry::collisions::aabbAabb(
         tree.getRoot().meta.aabb,
         tree.getNode(tree.getRoot().childNodeIds[1]).meta.aabb
      )
   );
}

TEST_CASE( "add two aabbs", "[aabb_tree]" )
{
   geometry::AabbTree tree(
      {1, {{1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f}}}
   );

   geometry::aabb_meta_t new_aabb = {0, {{0.5f, 0.5f, 0.5f}, {-0.75f, -0.75f, 0.f}}};

   geometry::growTree(new_aabb, tree);

   new_aabb.uid = 2;
   new_aabb.aabb.vertMin.Initialize(-2.f, -2.f, 1.f);
   new_aabb.aabb.vertMax.Initialize(-1.f, -1.5f, 1.5f);

   geometry::growTree(new_aabb, tree);

   REQUIRE( tree.size() == 4 );
   REQUIRE( tree.getRoot().childNodeIds[0] >= 0 );
   REQUIRE( tree.getRoot().childNodeIds[1] >= 0 );
   REQUIRE( tree.getRoot().childNodeIds[2] >= 0 );

   REQUIRE( tree.getNode(tree.getRoot().childNodeIds[0]).meta.uid >= 0 );
   REQUIRE( tree.getNode(tree.getRoot().childNodeIds[1]).meta.uid >= 0 );
   REQUIRE( tree.getNode(tree.getRoot().childNodeIds[2]).meta.uid >= 0 );
   REQUIRE( tree.getNode(tree.getRoot().childNodeIds[1]).meta.uid != tree.getNode(tree.getRoot().childNodeIds[0]).meta.uid );

}

// Verifies that adding four AABB's leads to one root node with four children
TEST_CASE( "add four aabbs", "[aabb_tree]" )
{
   geometry::AabbTree tree(
      {0, {{1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f}}}
   );

   geometry::aabb_meta_t new_aabb = {1, {{0.5f, 0.5f, 0.5f}, {-0.75f, -0.75f, 0.f}}};

   geometry::growTree(new_aabb, tree);

   new_aabb.uid = 2;
   new_aabb.aabb.vertMin.Initialize(-2.f, -2.f, 1.f);
   new_aabb.aabb.vertMax.Initialize(-1.f, -1.5f, 1.5f);

   geometry::growTree(new_aabb, tree);

   new_aabb.uid = 3;
   new_aabb.aabb.vertMin.Initialize(-2.f, -2.5f, -2.f);
   new_aabb.aabb.vertMax.Initialize(-1.f, -2.f, 0.5f);

   geometry::growTree(new_aabb, tree);

   REQUIRE( tree.size() == 5 );
   for (int i = 0; i < 4; ++i)
   {
      REQUIRE( tree.getRoot().childNodeIds[i] >= 0 );
   }

   REQUIRE( tree.getRoot().meta.uid < 0 );
}
