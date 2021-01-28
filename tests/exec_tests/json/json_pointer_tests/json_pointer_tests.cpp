#include "json_pointer.h"

#include "json_string.h"
#include "json_array.h"
#include "json_value.h"

#define CATCH_CONFIG_MAIN

#include <stdio.h>

#include "catch.hpp"

TEST_CASE( "empty pointer" )
{
   const char * json_pointer = "";

   json_array_t tokens;

   json_pointer_tokenize(json_pointer, &tokens);

   REQUIRE( tokens.size == 0 );

   // json_array_delete(&tokens);
   json_value_t temp_value;
   temp_value.value_type = JSON_ARRAY;
   temp_value.array = tokens;
   json_value_delete(&temp_value);
}

TEST_CASE( "one solidus" )
{
   const char * json_pointer = "/";

   json_array_t tokens;

   json_pointer_tokenize(json_pointer, &tokens);

   REQUIRE( tokens.size == 0 );

   // json_array_delete(&tokens);
   json_value_t temp_value;
   temp_value.value_type = JSON_ARRAY;
   temp_value.array = tokens;
   json_value_delete(&temp_value);
}

TEST_CASE( "multiple solidus" )
{
   const char * json_pointer = "/////";

   json_array_t tokens;

   json_pointer_tokenize(json_pointer, &tokens);

   REQUIRE( tokens.size == 0 );

   // json_array_delete(&tokens);
   json_value_t temp_value;
   temp_value.value_type = JSON_ARRAY;
   temp_value.array = tokens;
   json_value_delete(&temp_value);
}

TEST_CASE( "one item in pointer" )
{
   const char * json_pointer = "/stuff";

   json_array_t tokens;

   json_pointer_tokenize(json_pointer, &tokens);

   REQUIRE( tokens.size == 1 );

   // json_array_delete(&tokens);
   json_value_t temp_value;
   temp_value.value_type = JSON_ARRAY;
   temp_value.array = tokens;
   json_value_delete(&temp_value);
}

TEST_CASE( "two strings in pointer" )
{
   const char * json_pointer = "/doohickey@/things";

   json_array_t tokens;

   json_pointer_tokenize(json_pointer, &tokens);

   REQUIRE( tokens.size == 2 );

   // json_array_delete(&tokens);
   json_value_t temp_value;
   temp_value.value_type = JSON_ARRAY;
   temp_value.array = tokens;
   json_value_delete(&temp_value);
}

TEST_CASE( "one string and int in pointer" )
{
   const char * json_pointer = "/flibberty jibbit/0";

   json_array_t tokens;

   json_pointer_tokenize(json_pointer, &tokens);

   REQUIRE( tokens.size == 2 );

   // json_array_delete(&tokens);
   json_value_t temp_value;
   temp_value.value_type = JSON_ARRAY;
   temp_value.array = tokens;
   json_value_delete(&temp_value);
}

TEST_CASE( "multiple strings and ints in pointer" )
{
   const char * json_pointer = "/LeopalD McJingleheimer420x69/0/892/jeff foxworthy/dwarf fortress/library/author/43/-90";

   json_array_t tokens;

   json_pointer_tokenize(json_pointer, &tokens);

   REQUIRE( tokens.size == 9 );

   REQUIRE( json_pointer_token_type(&(tokens.vals[0].string)) == POINTER_TOKEN_STR);
   REQUIRE( json_pointer_token_type(&(tokens.vals[1].string)) == POINTER_TOKEN_INT);
   REQUIRE( json_pointer_token_type(&(tokens.vals[2].string)) == POINTER_TOKEN_INT);
   REQUIRE( json_pointer_token_type(&(tokens.vals[3].string)) == POINTER_TOKEN_STR);
   REQUIRE( json_pointer_token_type(&(tokens.vals[4].string)) == POINTER_TOKEN_STR);
   REQUIRE( json_pointer_token_type(&(tokens.vals[5].string)) == POINTER_TOKEN_STR);
   REQUIRE( json_pointer_token_type(&(tokens.vals[6].string)) == POINTER_TOKEN_STR);
   REQUIRE( json_pointer_token_type(&(tokens.vals[7].string)) == POINTER_TOKEN_INT);
   REQUIRE( json_pointer_token_type(&(tokens.vals[8].string)) == POINTER_TOKEN_STR);

   // json_array_delete(&tokens);
   json_value_t temp_value;
   temp_value.value_type = JSON_ARRAY;
   temp_value.array = tokens;
   json_value_delete(&temp_value);
}
