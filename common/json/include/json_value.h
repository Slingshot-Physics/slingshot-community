#ifndef JSON_VALUE_HEADER
#define JSON_VALUE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

void json_value_initialize(json_value_t * data);

// Allocates a new JSON value type and marks it as the child of the parent
// value. Returns a pointer to the child value type.
void json_value_allocate_type(
   json_value_t * value, K_VALUETYPE new_value_type
);

// Allocates a new blank value to the array portion of a JSON value node. The
// new value's parent is assigned to its array container.
json_value_t * json_value_append_new_blank_value_to_array(
   json_value_t * parent
);

void json_value_append_value_to_array(
   json_value_t * parent, json_value_t * new_val
);

// Allocates a new string key to the object portion of a JSON value node. The
// string value isn't assigned. The new key's parent is set to its container.
json_value_t * json_value_append_new_blank_key_to_object(
   json_value_t * parent
);

// Allocates a new blank value to the values portion of a JSON object node. The
// new value's parent is assigned to its object container.
json_value_t * json_value_append_new_blank_value_to_object(
   json_value_t * parent
);

void json_value_delete_object(json_value_t * node);

void json_value_delete_array(json_value_t * node);

// Recursively deletes this node and all of its children.
void json_value_delete(json_value_t * node);

// Allows the user to access elements of a JSON tree through a JSON pointer.
// JSON pointers are formatted as:
//
//    '/stuff/0/things'
//
// with the pattern that keys and indices are delimited by forward slashes
// (solidus).
int json_value_json_pointer_access(
   json_value_t * in_node,
   const char * json_pointer_str,
   json_value_t ** out_node
);

#ifdef __cplusplus
}
#endif

#endif
