#include "dynamic_array.hpp"
#include "slingshot_types.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

TEST_CASE( "construct", "[DynamicArray]" )
{
   DynamicArray<oy::types::rigidBody_t> rb_array(32);

   REQUIRE(rb_array.capacity() == 32);
   REQUIRE(rb_array.size() == 0);
}

TEST_CASE( "append within capacity", "[DynamicArray]" )
{
   DynamicArray<oy::types::rigidBody_t> rb_array(32);

   REQUIRE(rb_array.capacity() == 32);
   REQUIRE(rb_array.size() == 0);

   oy::types::rigidBody_t temp_body;
   temp_body.mass = 0.f;

   for (unsigned int i = 0; i < 12; ++i)
   {
      rb_array.append(temp_body);
      temp_body.mass += 1;
   }

   REQUIRE(rb_array.capacity() == 32);
   REQUIRE(rb_array.size() == 12);

   for (unsigned int i = 0; i < 12; ++i)
   {
      REQUIRE(rb_array[i].mass == i);
   }

}

TEST_CASE( "out of bounds access to empty array via at", "[DynamicArray]" )
{
   DynamicArray<oy::types::rigidBody_t> rb_array(32);
   bool exception_thrown = false;
   try
   {
      rb_array.at(0);
   }
   catch(const std::exception& e)
   {
      std::cerr << e.what() << '\n';
      exception_thrown = true;
   }
   
   REQUIRE( exception_thrown );
}

TEST_CASE( "in bounds access to non-empty array via at", "[DynamicArray]" )
{
   DynamicArray<int> array(32);
   array.push_back(-123);
   array.push_back(-77);
   array.push_back(7);
   array.push_back(22);
   array.push_back(40);
   bool exception_thrown = false;
   try
   {
      REQUIRE( array.at(0) == -123 );
      REQUIRE( array.at(1) == -77 );
      REQUIRE( array.at(2) == 7 );
      REQUIRE( array.at(3) == 22 );
   }
   catch(const std::exception& e)
   {
      std::cerr << e.what() << '\n';
      exception_thrown = true;
   }
   
   REQUIRE( !exception_thrown );
}

TEST_CASE( "out of bounds access to non-empty array via at", "[DynamicArray]" )
{
   DynamicArray<oy::types::rigidBody_t> rb_array(32);

   for (int i = 0; i < 100; ++i)
   {
      rb_array.push_back(oy::types::rigidBody_t{});
   }

   bool exception_thrown = false;
   try
   {
      rb_array.at(100);
   }
   catch(const std::exception& e)
   {
      std::cerr << e.what() << '\n';
      exception_thrown = true;
   }
   
   REQUIRE( exception_thrown );
}

TEST_CASE( "append beyond capacity", "[DynamicArray]" )
{
   DynamicArray<oy::types::rigidBody_t> rb_array(32);

   const unsigned int NUM_ELEMENTS = 42;

   REQUIRE(rb_array.capacity() == 32);
   REQUIRE(rb_array.size() == 0);

   oy::types::rigidBody_t temp_body;
   temp_body.mass = 0.f;

   for (unsigned int i = 0; i < NUM_ELEMENTS; ++i)
   {
      rb_array.append(temp_body);
      temp_body.mass += 1;
   }

   REQUIRE(rb_array.capacity() == 64);
   REQUIRE(rb_array.size() == NUM_ELEMENTS);

   for (unsigned int i = 0; i < NUM_ELEMENTS; ++i)
   {
      REQUIRE(rb_array[i].mass == i);
   }
}

TEST_CASE( "append beyond capacity and clear", "[DynamicArray]" )
{
   DynamicArray<oy::types::rigidBody_t> rb_array(32);

   const unsigned int NUM_ELEMENTS = 42;

   REQUIRE(rb_array.capacity() == 32);
   REQUIRE(rb_array.size() == 0);

   oy::types::rigidBody_t temp_body;
   temp_body.mass = 0.f;

   for (unsigned int i = 0; i < NUM_ELEMENTS; ++i)
   {
      rb_array.append(temp_body);
      temp_body.mass += 1;
   }

   REQUIRE(rb_array.capacity() == 64);
   REQUIRE(rb_array.size() == NUM_ELEMENTS);

   for (unsigned int i = 0; i < NUM_ELEMENTS; ++i)
   {
      REQUIRE(rb_array[i].mass == i);
   }

   rb_array.clear();

   REQUIRE(rb_array.size() == 0);
   REQUIRE(rb_array.capacity() == 64);

   rb_array.append(temp_body);

   REQUIRE(rb_array.size() == 1);
}

TEST_CASE( "one pop", "[DynamicArray]")
{
   DynamicArray<int> arr(32);

   for (int i = 0; i < 10; ++i)
   {
      arr.push_back(i);
   }

   int val = arr.pop(5);
   REQUIRE( val == 5);

   REQUIRE(arr.size() == 9);

   REQUIRE(arr[5] == 6);
   REQUIRE(arr[6] == 7);
   REQUIRE(arr[7] == 8);
   REQUIRE(arr[8] == 9);
}

TEST_CASE( "multiple pops", "[DynamicArray]")
{
   DynamicArray<int> arr(32);

   for (int i = 0; i < 10; ++i)
   {
      arr.push_back(i);
   }

   int val = arr.pop(0);
   REQUIRE( val == 0);
   REQUIRE( arr.size() == 9 );

   val = arr.pop(5);
   REQUIRE( val == 6 );
   REQUIRE( arr.size() == 8 );

   REQUIRE( arr[5] == 7 );
   REQUIRE( arr[6] == 8 );
   REQUIRE( arr[7] == 9 );
}

TEST_CASE( "pop all", "[DynamicArray]")
{
   DynamicArray<int> arr(32);

   for (int i = 0; i < 10; ++i)
   {
      arr.push_back(i);
   }

   REQUIRE( arr.size() == 10 );

   for (int i = 0; i < 10; ++i)
   {
      int val = arr.pop(0);
      REQUIRE( val == i );
      REQUIRE( arr.size() == (10 - (i + 1)) );
   }
}

TEST_CASE( "get end of small array", "[DynamicArray]")
{
   DynamicArray<int> arr(4);

   DynamicArray<int>::const_iterator const_end = arr.end();

   DynamicArray<int>::iterator end = arr.end();

   REQUIRE( true );
}
