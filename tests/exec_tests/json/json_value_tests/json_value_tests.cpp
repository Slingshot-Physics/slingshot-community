#include "json_deserialize.h"

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


TEST_CASE( "tricky json pointer object access" )
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

   json_value_t * pointer_output = nullptr;
   int pointer_access = json_value_json_pointer_access(&tree, "/blub/burgle", &pointer_output);
   REQUIRE( pointer_access > 0 );

   printf("pointer output? %p\n", pointer_output);

   REQUIRE( pointer_output != nullptr );
   if (pointer_output == nullptr)
   {
      json_string_delete(&temp_str);
      json_value_delete(&tree);
      return;
   }
   REQUIRE( pointer_output->value_type == JSON_OBJECT );
   printf("output key: %s\n", pointer_output->object.keys[0].string.buffer);

   json_string_delete(&temp_str);
   json_value_delete(&tree);
}

TEST_CASE( "tricky json pointer array access" )
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

   json_value_t * pointer_output = nullptr;

   int pointer_access = json_value_json_pointer_access(&tree, "/blub/stifle/1", &pointer_output);
   REQUIRE( pointer_access > 0 );

   REQUIRE( pointer_output != nullptr );
   if (pointer_output == nullptr)
   {
      json_string_delete(&temp_str);
      json_value_delete(&tree);
      return;
   }
   REQUIRE( pointer_output->value_type == JSON_FLOAT_NUMBER );
   printf("output value: %f\n", pointer_output->fnum);

   json_string_delete(&temp_str);
   json_value_delete(&tree);
}
