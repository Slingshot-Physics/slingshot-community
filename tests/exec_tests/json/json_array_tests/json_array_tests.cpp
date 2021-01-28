#include "json_types.h"

#include "json_array.h"
#include "json_string.h"
#include "json_value.h"

#define CATCH_CONFIG_MAIN

#include <stdio.h>
#include <string.h>

#include "catch.hpp"

TEST_CASE( "append int to array" )
{
   json_array_t array;
   json_array_initialize(&array);

   REQUIRE( array.size == 0 );

   json_value_t new_val;
   new_val.value_type = JSON_INT_NUMBER;
   new_val.inum = 5;

   json_array_append(&array, &new_val);

   REQUIRE( array.size == 1 );

   REQUIRE( array.vals[0].inum == 5 );

   json_array_delete(&array);
}

TEST_CASE( "append float and int to array" )
{
   json_array_t array;
   json_array_initialize(&array);

   REQUIRE( array.size == 0 );

   json_value_t new_ival;
   new_ival.value_type = JSON_INT_NUMBER;
   new_ival.inum = 5;

   json_array_append(&array, &new_ival);

   REQUIRE( array.size == 1 );

   REQUIRE( array.vals[0].inum == 5 );

   json_value_t new_fval;
   new_fval.value_type = JSON_FLOAT_NUMBER;
   new_fval.fnum = -3.14159265;

   json_array_append(&array, &new_fval);

   REQUIRE( array.size == 2 );

   REQUIRE( (array.vals[1].fnum - -3.14159265) < 1e-5f );

   json_array_delete(&array);
}

TEST_CASE( "append float, int, string to array" )
{
   json_array_t array;
   json_array_initialize(&array);

   REQUIRE( array.size == 0 );
   json_value_t new_ival;
   new_ival.value_type = JSON_INT_NUMBER;
   new_ival.inum = 75;

   json_array_append(&array, &new_ival);

   REQUIRE( array.size == 1 );
   REQUIRE( array.vals[0].inum == 75 );

   json_value_t new_fval;
   new_fval.value_type=JSON_FLOAT_NUMBER;
   new_fval.fnum=-3.14159265;

   json_array_append(&array, &new_fval);

   REQUIRE( array.size == 2 );
   REQUIRE( (array.vals[1].fnum - -3.14159265) < 1e-5f );

   json_value_t new_sval;
   new_sval.value_type=JSON_STRING;
   json_string_initialize(&new_sval.string);
   json_string_allocate(&new_sval.string);

   json_string_make(&new_sval.string, "boston dynamics");

   json_array_append(&array, &new_sval);

   REQUIRE( array.size == 3 );
   REQUIRE( strncmp(array.vals[2].string.buffer, "boston dynamics", 15) == 0);

   json_value_t encapsulator;
   encapsulator.value_type=JSON_ARRAY;
   encapsulator.array=array;

   json_value_delete_array(&encapsulator);
}
