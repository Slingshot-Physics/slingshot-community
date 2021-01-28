#ifndef JSON_ARRAY_HEADER
#define JSON_ARRAY_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

void json_array_initialize(json_array_t * data);

void json_array_delete(json_array_t * data);

void json_array_allocate(json_array_t * data);

void json_array_increase_capacity(json_array_t * data);

void json_array_append(json_array_t * data, const json_value_t * element);

json_value_t json_array_pop_last(json_array_t * data);

#ifdef __cplusplus
}
#endif

#endif
