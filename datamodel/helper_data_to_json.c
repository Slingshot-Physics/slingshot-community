#ifdef __cplusplus
extern "C"
{
#endif

#include "helper_data_to_json.h"

#include "json_array.h"
#include "json_object.h"
#include "json_string.h"
#include "json_value.h"

#include <stdlib.h>

// Adds a key value pair for "_typename": typename to the current node.
int add_typename(json_value_t * node, const char * type_name)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * typename_key = json_value_append_new_blank_key_to_object(node);
   json_value_allocate_type(typename_key, JSON_STRING);
   json_string_make(&(typename_key->string), "_typename");

   json_value_t * typename_val = json_value_append_new_blank_value_to_object(node);
   json_value_allocate_type(typename_val, JSON_STRING);
   json_string_make(&(typename_val->string), type_name);

   return 1;
}

json_value_t * add_field(
   json_value_t * node, const char * field_name, K_VALUETYPE field_type
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return NULL;
   }

   json_value_t * field_name_key = json_value_append_new_blank_key_to_object(node);
   json_value_allocate_type(field_name_key, JSON_STRING);
   json_string_make(&(field_name_key->string), field_name);

   json_value_t * field_name_val = json_value_append_new_blank_value_to_object(node);
   json_value_allocate_type(field_name_val, field_type);

   return field_name_val;
}

json_value_t * get_field_by_name(
   json_value_t * node, const char * field_name
)
{
   if (node == NULL || field_name == NULL || node->value_type != JSON_OBJECT)
   {
      return NULL;
   }

   json_string_t temp_string;
   json_string_initialize(&temp_string);
   json_string_allocate(&temp_string);
   json_string_make(&temp_string, field_name);

   json_value_t * field_value = json_object_find_by_key(
      &(node->object), &temp_string
   );

   json_string_delete(&temp_string);

   return field_value;
}

int add_optional_dynamic_object_array_field(
   json_value_t * node,
   const char * array_field_name,
   unsigned int length,
   int max_length,
   data_to_json_f jsonifier,
   const void * data_array,
   unsigned int element_size
)
{
   if (length == 0)
   {
      return 1;
   }

   return add_filled_dynamic_object_array_field(
      node,
      array_field_name,
      length,
      max_length,
      jsonifier,
      data_array,
      element_size
   );
}

int add_dynamic_object_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   int max_length
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_name_val = add_field(node, field_name, JSON_ARRAY);
   if (field_name_val == NULL)
   {
      return 0;
   }

   for (unsigned int i = 0; i < length && i < max_length; ++i)
   {
      json_value_t temp_val = {.value_type = JSON_OBJECT};
      json_value_append_value_to_array(field_name_val, &temp_val);
      json_value_allocate_type(&(field_name_val->array.vals[i]), JSON_OBJECT);
   }

   return 1;
}

int add_filled_dynamic_object_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   int max_length,
   data_to_json_f jsonifier,
   const void * data_array,
   unsigned int element_size
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_name_val = add_field(node, field_name, JSON_ARRAY);
   if (field_name_val == NULL)
   {
      return 0;
   }

   for (unsigned int i = 0; i < length && i < max_length; ++i)
   {
      json_value_t temp_val = {.value_type = JSON_OBJECT};
      json_value_append_value_to_array(field_name_val, &temp_val);
      json_value_t * temp_elem = &(field_name_val->array.vals[i]);
      json_value_allocate_type(temp_elem, JSON_OBJECT);
      jsonifier(temp_elem, data_array + i * element_size);
   }

   return 1;
}

int add_fixed_float_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   const float * arr
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_name_val = add_field(node, field_name, JSON_ARRAY);
   if (field_name_val == NULL)
   {
      return 0;
   }

   for (unsigned int i = 0; i < length; ++i)
   {
      json_value_t temp_val = {
         .value_type=JSON_FLOAT_NUMBER,
         .fnum = arr[i]
      };
      json_value_append_value_to_array(field_name_val, &temp_val);
   }

   return 1;
}

int add_fixed_int_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   const int * arr
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_name_val = add_field(node, field_name, JSON_ARRAY);
   if (field_name_val == NULL)
   {
      return 0;
   }

   for (unsigned int i = 0; i < length; ++i)
   {
      json_value_t temp_val = {
         .value_type=JSON_INT_NUMBER,
         .inum = arr[i]
      };
      json_value_append_value_to_array(field_name_val, &temp_val);
   }

   return 1;
}

int add_fixed_uint_array_field(
   json_value_t * node,
   const char * field_name,
   unsigned int length,
   const unsigned int * arr
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_name_val = add_field(node, field_name, JSON_ARRAY);
   if (field_name_val == NULL)
   {
      return 0;
   }

   for (unsigned int i = 0; i < length; ++i)
   {
      json_value_t temp_val = {
         .value_type=JSON_INT_NUMBER,
         .inum = arr[i]
      };
      json_value_append_value_to_array(field_name_val, &temp_val);
   }

   return 1;
}

int add_float_field(
   json_value_t * node, const char * field_name, float val
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_value = add_field(node, field_name, JSON_FLOAT_NUMBER);
   if (field_value == NULL)
   {
      return 0;
   }

   field_value->fnum = val;

   return 1;
}

int add_int_field(
   json_value_t * node, const char * field_name, int val
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_value = add_field(node, field_name, JSON_INT_NUMBER);
   if (field_value == NULL)
   {
      return 0;
   }

   field_value->inum = val;

   return 1;
}

int add_uint_field(
   json_value_t * node, const char * field_name, unsigned int val
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_value = add_field(node, field_name, JSON_INT_NUMBER);
   if (field_value == NULL)
   {
      return 0;
   }

   field_value->inum = (int )val;

   return 1;
}

int add_string_field(
   json_value_t * node,
   const char * field_name,
   const char * in_str
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * field_value = add_field(node, field_name, JSON_STRING);
   if (field_value == NULL)
   {
      return 0;
   }

   json_string_make(&(field_value->string), in_str);

   return 1;
}

int add_object_field(
   json_value_t * node,
   const char * field_name,
   data_to_json_f jsonifier,
   const void * data
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   json_value_t * obj_field = add_field(node, field_name, JSON_OBJECT);
   if (obj_field == NULL)
   {
      return 0;
   }
   if (!jsonifier(obj_field, data))
   {
      return 0;
   }

   return 1;
}

#ifdef __cplusplus
}
#endif
