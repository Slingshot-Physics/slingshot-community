#ifndef JSON_OBJECT_HEADER
#define JSON_OBJECT_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

void json_object_initialize(json_object_t * data);

void json_object_delete(json_object_t * data);

void json_object_allocate(json_object_t * data);

void json_object_increase_capacity(json_object_t * data);

// Inserts a (key, value) pair into the object.
void json_object_append(
   json_object_t * data, const json_value_t * key, const json_value_t * value
);

// This should be used carefully. This adds a key and a blank value and
// increases the size of the object by 1. It's assumed that the next call will
// add a value to the object
void json_object_append_key(json_object_t * data, const json_value_t * key);

// Assumes that a new key has been appended to the object already and that the
// size of the object has been increased. Assigns the key at 'size - 1' to
// 'value'.
void json_object_append_value(
   json_object_t * data, const json_value_t * value
);

// Given a JSON object type and a string key, loop through the object and find
// the value corresponding to the given key. If a value exists for the key, a
// pointer to the value is returned. If no value exists for the key, a NULL
// pointer is returned.
json_value_t * json_object_find_by_key(
   const json_object_t * data, const json_string_t * key
);

#ifdef __cplusplus
}
#endif

#endif
