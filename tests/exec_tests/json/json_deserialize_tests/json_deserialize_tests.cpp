#include "json_deserialize.h"
#include "json_serialize.h"
#include "json_verify.h"

#include "json_string.h"
#include "json_value.h"

#define CATCH_CONFIG_MAIN

#include <cstddef>
#include <stdio.h>

#include "catch.hpp"

// formerly known as test.json
const char * big_tricky_json_str = 
"{   " // sneaky whitespaces
   "   \"blub\":"
   "   {    "
      " \"burgle\":   "
      "  {"
         "\"bingle\":"
         "{"
            "\"dingle\":"
            "{"
               "\"dongle\":"
               "{"
                  "\"dabble\":[2.8]"
               "}"
            "}"
         "}"
      "},"
      "\"stifle\":[3,5.2]"
   "},"
   "\"thing\": ["
      "1,2,"
      "{"
         "\"sneaky\": ["
            "\"biznatch\", \"cross-hatch\", \"hatch-back\", \"back hoe\","
            "\"hoedown\", \"downfall\""
         "]"
      "}"
   "],"
   "\"stuff\":45,"
   "\"skrug\":["
      "[2,4.52,7e3,[5,2]],"
      "[2,1,-8]"
   "]"
"}";

// formerly known as test2.json
const char * small_json_str = "{"
   "\"thing\": 10.2,"
   "\"stuff\": [1, 2, 3],"
   "\"big_\\n\\r\\\"things\": ["
      "[1, 2, 3],"
      "[4, 5, 6, 7],"
      "{"
         "\"boogers\": {"
            "\"nose\\t\\\"\\"" : 75.3e-1"
         "},"
         "\"phlegm\": \"throat\""
      "}"
   "],"
   "\"small_things\": \"fois gras\","
   "\"vector3\":"
   "{"
      "\"type\": \"float\","
      "\"vals\": [6, 7, 8]"
   "}"
"}";


// Verifies that a very simple JSON string dictionary can be deserialized.
TEST_CASE( "basic_json_deserializer_test_1" )
{
   // const char * basic_json = "{\"basic\":\"test\"}";
   json_string_t temp_str;
   json_string_initialize(&temp_str);
   json_string_allocate(&temp_str);
   
   // json_string_make(&temp_str, basic_json);
   json_string_make(&temp_str, "{\"basic\":\"test\"}");

   json_value_t tree;
   json_value_initialize(&tree);

   json_deserialize_str(&temp_str, &tree);

   REQUIRE( tree.value_type == JSON_OBJECT );

   REQUIRE( tree.object.size == 1 );
   REQUIRE( strncmp(tree.object.keys[0].string.buffer, "basic", 5) == 0 );
   REQUIRE( tree.object.values[0].value_type == JSON_STRING );
   REQUIRE( strncmp(tree.object.values[0].string.buffer, "test", 4) == 0 );

   json_string_delete(&temp_str);
   json_value_delete(&tree);

   REQUIRE( temp_str.buffer == nullptr );

   REQUIRE( tree.object.keys == nullptr );
   REQUIRE( tree.object.values == nullptr );
}

TEST_CASE( "basic_json_deserializer_test_2" )
{
   json_string_t temp_str;
   json_string_initialize(&temp_str);
   json_string_allocate(&temp_str);
   
   json_string_make(&temp_str, "{\"basic\":2}");

   json_value_t tree;
   json_value_initialize(&tree);

   json_deserialize_str(&temp_str, &tree);

   REQUIRE( tree.value_type == JSON_OBJECT );

   REQUIRE( tree.object.size == 1 );
   REQUIRE( strncmp(tree.object.keys[0].string.buffer, "basic", 5) == 0 );
   REQUIRE( tree.object.values[0].value_type == JSON_INT_NUMBER );
   REQUIRE( tree.object.values[0].inum == 2 );

   json_string_delete(&temp_str);
   json_value_delete(&tree);

   REQUIRE( temp_str.buffer == nullptr );

   REQUIRE( tree.object.keys == nullptr );
   REQUIRE( tree.object.values == nullptr );
}

// Attempts to deserialize a more difficult json string.
TEST_CASE( "advanced_json_deserializer_test_1" )
{
   json_string_t temp_str;
   json_string_initialize(&temp_str);
   json_string_allocate(&temp_str);

   if (!json_string_make(&temp_str, big_tricky_json_str))
   {
      json_string_delete(&temp_str);
      REQUIRE( 0 );
      return;
   }

   json_value_t tree;
   json_value_initialize(&tree);

   json_deserialize_str(&temp_str, &tree);

   REQUIRE( tree.value_type == JSON_OBJECT );
   REQUIRE( strncmp(tree.object.keys[0].string.buffer, "blub", 4) == 0 );
   REQUIRE( tree.object.values[0].value_type == JSON_OBJECT );
   REQUIRE( strncmp(tree.object.values[0].object.keys[0].string.buffer, "burgle", 6) == 0 );
   REQUIRE( tree.object.values[0].object.values[0].value_type == JSON_OBJECT );

   json_string_t reserialized_str;

   json_serialize_to_str(&tree, &reserialized_str);

   printf("reserialized str: %s\n", reserialized_str.buffer);

   REQUIRE(reserialized_str.size > 0);

   json_string_delete(&reserialized_str);
   json_string_delete(&temp_str);
   json_value_delete(&tree);
}

TEST_CASE( "deserialize empty array" )
{
   const char * empty_arr = "{\"things\": []}";

   json_string_t temp_str;
   json_string_initialize(&temp_str);
   json_string_allocate(&temp_str);

   if (!json_string_make(&temp_str, empty_arr))
   {
      json_string_delete(&temp_str);
      REQUIRE( 0 );
      return;
   }

   json_value_t tree;
   json_value_initialize(&tree);

   json_deserialize_str(&temp_str, &tree);

   REQUIRE( tree.value_type == JSON_OBJECT );
   REQUIRE( tree.object.size == 1 );
   REQUIRE( tree.object.keys[0].value_type == JSON_STRING );
   REQUIRE( tree.object.values[0].value_type == JSON_ARRAY );
   printf("%d\n", tree.object.values[0].array.vals[0].value_type);
   REQUIRE( tree.object.values[0].array.size == 0 );
}
