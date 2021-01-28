#ifndef HELPER_DATA_TO_JSON_HEADER
#define HELPER_DATA_TO_JSON_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef int (*data_to_json_f)(json_value_t * node, const void * data);

// Adds a key value pair for "_typename": typename to the current node.
int add_typename(json_value_t * node, const char * type_name);

// This function allocates a new key/value pair for the JSON_OBJECT 'node'.
// A new string is added to the JSON_OBJECT. A new value is also added to the
// JSON_OBJECT, with the value being allocated based on its field type.
json_value_t * add_field(
   json_value_t * node, const char * field_name, K_VALUETYPE field_type
);

// Searches a JSON_OBJECT's keys for a string that matches 'field_name'.
// Returns a pointer to the value if a match is found, NULL otherwise.
// Similar to node[field_name].
json_value_t * get_field_by_name(
   json_value_t * node, const char * field_name
);

int add_optional_dynamic_object_array_field(
   json_value_t * node,
   const char * array_field_name,
   unsigned int length,
   int max_length,
   data_to_json_f jsonifier,
   const void * data_array,
   unsigned int element_size
);

int add_dynamic_object_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   int max_length
);

int add_filled_dynamic_object_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   int max_length,
   data_to_json_f jsonifier,
   const void * data_array,
   unsigned int element_size
);

int add_fixed_float_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   const float * arr
);

int add_fixed_int_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   const int * arr
);

// Adds an array entry to the tree. 'field_name' is the name of the struct
// member associated with the unsigned integral array.
int add_fixed_uint_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   const unsigned int * arr
);

// Adds a float entry to the tree where the 'field_name' is the name of the
// float field whose data is in 'val'.
int add_float_field(
   json_value_t * node, const char * field_name, float val
);

// Adds an int entry to the tree where the 'field_name' is the name of the int
// field whose data is in 'val'.
int add_int_field(
   json_value_t * node, const char * field_name, int val
);

// Adds an unsigned int entry to the tree where the 'field_name' is the name of
// the unsigned int field whose data is in 'val'.
int add_uint_field(
   json_value_t * node, const char * field_name, unsigned int val
);

// Adds a string entry to the tree where the 'field_name' is the name of the
// string field whose data is contained in 'in_str'. The field_name is added as
// a key to the existing tree, and the value is a json_string containing the
// characters of 'in_str'.
int add_string_field(
   json_value_t * node,
   const char * field_name,
   const char * in_str
);

// Adds an object entry to the tree where the 'field_name' is the name of the
// struct field whose data is contained in 'data'. The field_name is added as
// a key to the existing tree, and the json-ification of 'data' is added as a
// value attached to the field_name key.
int add_object_field(
   json_value_t * node,
   const char * field_name,
   data_to_json_f jsonifier,
   const void * data
);

#ifdef __cplusplus
}
#endif

#endif
