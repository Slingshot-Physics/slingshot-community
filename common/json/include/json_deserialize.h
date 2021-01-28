#ifndef JSON_DESERIALIZE_HEADER
#define JSON_DESERIALIZE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include <stdio.h>
#include <stdlib.h>

struct json_parse_state_s;

typedef void (*parse_f)(struct json_parse_state_s * state, char c);

typedef struct json_parse_state_s
{
   json_value_t * node;
   parse_f parser;
   json_string_t buffer;
} json_parse_state_t;

// Initializes and allocates a copy of the json string in 'in_json' that is
// free of whitespace padding. Note that this function will initialize and
// allocate 'out_json'.
void json_strip(
   const json_string_t * in_json, json_string_t * out_json
);

void json_merge_buffer_to_value(
   const json_string_t * buffer, json_value_t * val
);

parse_f json_assign_parser(const json_value_t * state);

void json_parse_value(json_parse_state_t * state, char c);

void json_parse_string(json_parse_state_t * state, char c);

void json_parse_number(json_parse_state_t * state, char c);

void json_parse_fraction(json_parse_state_t * state, char c);

void json_parse_exponent(json_parse_state_t * state, char c);

void json_parse_object(json_parse_state_t * state, char c);

void json_parse_array(json_parse_state_t * state, char c);

int json_deserialize_str(json_string_t * json_data, json_value_t * root);

int json_deserialize_file(FILE * file_ptr, json_value_t * root);

#ifdef __cplusplus
}
#endif

#endif
