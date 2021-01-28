#include "json_deserialize.h"
#include "json_serialize.h"
#include "json_verify.h"

#include "json_string.h"
#include "json_value.h"

#define CATCH_CONFIG_MAIN

#include <stdio.h>

#include "catch.hpp"

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

TEST_CASE( "basic good JSON is good" )
{
   json_string_t temp_str;
   json_string_initialize(&temp_str);
   json_string_allocate(&temp_str);
   
   json_string_make(&temp_str, "{\"basic\":2}");

   REQUIRE (json_verify_basic(&temp_str) > 0);

   json_string_delete(&temp_str);
}

TEST_CASE( "basic bad JSON is bad" )
{
   json_string_t temp_str;
   json_string_initialize(&temp_str);
   json_string_allocate(&temp_str);
   
   json_string_make(&temp_str, "{\"basic\":2");

   REQUIRE (json_verify_basic(&temp_str) < 0);

   json_string_delete(&temp_str);
}

TEST_CASE( "advanced good JSON is good" )
{
   printf("good, but tricky, json:\n%s\n", small_json_str);

   json_string_t temp_str;
   json_string_initialize(&temp_str);
   json_string_allocate(&temp_str);
   
   json_string_make(&temp_str, small_json_str);

   REQUIRE (json_verify_basic(&temp_str) < 0);

   json_string_delete(&temp_str);
}
