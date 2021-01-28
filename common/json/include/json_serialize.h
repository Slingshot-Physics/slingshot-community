#ifndef JSON_SERIALIZE_HEADER
#define JSON_SERIALIZE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include <stdio.h>

void json_serialize_string(
   const json_value_t * node, int level, json_string_t * out
);

void json_serialize_object(
   const json_value_t * node, int level, json_string_t * out
);

void json_serialize_array(
   const json_value_t * node, int level, json_string_t * out
);

void json_serialize_value(
   const json_value_t * node, int level, json_string_t * out
);

void json_serialize_to_str(const json_value_t * value, json_string_t * out);

void json_serialize_to_file(const json_value_t * value, FILE * file_ptr);

#ifdef __cplusplus
}
#endif

#endif
