#ifndef HELPER_JSON_TO_DATA_HEADER
#define HELPER_JSON_TO_DATA_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "json_object.h"
#include "json_string.h"
#include "json_value.h"

#include <string.h>
#include <stdlib.h>

typedef int (*data_from_json_f)(const json_value_t * node, void * data);

// Returns 1 if the typename matches, 0 otherwise.
int verify_typename(const json_value_t * node, const char * type_name);

const json_value_t * get_const_field_by_name(
   const json_value_t * node, const char * field_name
);

// If the 'array_field_name' is present in the JSON node, then:
//    1. the 'data_array' is allocated 
//    2. the data in the JSON node is structified and copied into the
//       'data_array'
//    3. the 'counter' is set to the number of elements in the JSON array.
//
// If the 'array_field_name' is not present, then:
//    1. the 'counter' is set to zero
//    2. the 'data_array' is set to NULL
int copy_optional_dynamic_object_array_field(
   const json_value_t * node,
   const char * array_field_name,
   int max_length,
   unsigned int * counter,
   data_from_json_f structifier,
   void ** data_array,
   unsigned int element_size
);

int copy_filled_dynamic_object_array_field(
   const json_value_t * node,
   const char * field_name,
   unsigned int length,
   int max_length,
   data_from_json_f structifier,
   void * data_array,
   unsigned int element_size
);

int copy_fixed_float_array_field(
   const json_value_t * node,
   const char * field_name,
   unsigned int length,
   float * arr
);

int copy_fixed_int_array_field(
   const json_value_t * node,
   const char * field_name,
   unsigned int length,
   int * arr
);

int copy_fixed_uint_array_field(
   const json_value_t * node,
   const char * field_name,
   unsigned int length,
   unsigned int * arr
);

int copy_float_field(
   const json_value_t * node, const char * field_name, float * fnum
);

int copy_int_field(
   const json_value_t * node, const char * field_name, int * inum
);

int copy_uint_field(
   const json_value_t * node, const char * field_name, unsigned int * uinum
);

int copy_string_field(
   const json_value_t * node,
   const char * field_name,
   char * out_str,
   unsigned int max_length
);

int copy_object_field(
   const json_value_t * node,
   const char * field_name,
   data_from_json_f structifier,
   void * data
);

#ifdef __cplusplus
}
#endif

#endif
