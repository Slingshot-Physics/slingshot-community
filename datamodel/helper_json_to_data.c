#ifdef __cplusplus
extern "C"
{
#endif

#include "helper_json_to_data.h"

#include "json_object.h"
#include "json_string.h"
#include "json_value.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Returns 1 if the typename matches, 0 otherwise.
int verify_typename(const json_value_t * node, const char * type_name)
{
   if (node->value_type != JSON_OBJECT)
   {
      printf("not an object\n");
      return 0;
   }

   json_string_t temp_type_name_str;
   json_string_initialize(&temp_type_name_str);
   json_string_allocate(&temp_type_name_str);
   json_string_make(&temp_type_name_str, "_typename");

   json_value_t * type_name_value = json_object_find_by_key(
      &(node->object), &temp_type_name_str
   );

   json_string_delete(&temp_type_name_str);

   if (type_name_value == NULL || type_name_value->value_type != JSON_STRING)
   {
      printf("couldn't find key %*s\n", temp_type_name_str.size, temp_type_name_str.buffer);
      return 0;
   }

   if (
      strncmp(
         type_name_value->string.buffer,
         type_name,
         type_name_value->string.size
      ) != 0
   )
   {
      printf("typename check failed between provided %s and %*s\n", type_name, type_name_value->string.size, type_name_value->string.buffer);
      return 0;
   }

   return 1;
}

const json_value_t * get_const_field_by_name(
   const json_value_t * node, const char * field_name
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

   const json_value_t * field_value = json_object_find_by_key(
      &(node->object), &temp_string
   );

   json_string_delete(&temp_string);

   return field_value;
}

int copy_optional_dynamic_object_array_field(
   const json_value_t * node,
   const char * array_field_name,
   int max_length,
   unsigned int * counter,
   data_from_json_f structifier,
   void ** data_array,
   unsigned int element_size
)
{
   const json_value_t * array_field = get_const_field_by_name(
      node, array_field_name
   );

   if (array_field == NULL || array_field->array.size == 0)
   {
      *data_array = NULL;
      *counter = 0;

      return 1;
   }

   *counter = array_field->array.size;

   unsigned int requested_array_size = element_size * (*counter);
   unsigned int max_array_size = (unsigned int)max_length * element_size;

   unsigned int allocated_array_size = (
      (requested_array_size < max_array_size)
      ? requested_array_size
      : max_array_size
   );

   *data_array = malloc(allocated_array_size);

   return copy_filled_dynamic_object_array_field(
      node,
      array_field_name,
      *counter,
      max_length,
      structifier,
      *data_array,
      element_size
   );
}

int copy_filled_dynamic_object_array_field(
   const json_value_t * node,
   const char * field_name,
   unsigned int length,
   int max_length,
   data_from_json_f structifier,
   void * data_array,
   unsigned int element_size
)
{
   if (node->value_type != JSON_OBJECT)
   {
      printf("trying to copy filled dynamic object array - node type isn't object\n");
      return 0;
   }

   const json_value_t * field_value = get_const_field_by_name(node, field_name);

   if (
      field_value == NULL ||
      field_value->value_type != JSON_ARRAY ||
      field_value->array.size < length
   )
   {
      if (field_value == NULL)
      {
         printf("null field for string %s\n", field_name);   
      }
      else
      {
         printf("null field: %d\nwrong type: %d\nbad length: %d\n", (field_value == NULL), (field_value->value_type != JSON_ARRAY), (field_value->array.size < length));
      }
      return 0;
   }

   for (unsigned int i = 0; i < length && i < max_length; ++i)
   {
      const json_value_t * temp_elem = &(field_value->array.vals[i]);
      if (!structifier(temp_elem, data_array + i * element_size))
      {
         return 0;
      }
   }

   return 1;
}

int copy_fixed_float_array_field(
   const json_value_t * node,
   const char * field_name,
   unsigned int length,
   float * arr
)
{
   if (arr == NULL)
   {
      printf("output array is null\n");
      return 0;
   }

   const json_value_t * field_value = get_const_field_by_name(node, field_name);

   if (
      field_value == NULL ||
      field_value->value_type != JSON_ARRAY ||
      field_value->array.size < length
   )
   {
      printf("field value failure: %p\n", field_value);
      if (field_value != NULL)
      {
         printf("field value failure: %d, %u\n", field_value->value_type, field_value->array.size);
      }
      return 0;
   }

   for (unsigned int i = 0; i < length; ++i)
   {
      if (field_value->array.vals[i].value_type == JSON_INT_NUMBER)
      {
         arr[i] = (float )field_value->array.vals[i].inum;
      }
      else
      {
         arr[i] = field_value->array.vals[i].fnum;
      }
   }

   return 1;
}

int copy_fixed_int_array_field(
   const json_value_t * node,
   const char * field_name,
   unsigned int length,
   int * arr
)
{
   if (arr == NULL)
   {
      return 0;
   }

   const json_value_t * field_value = get_const_field_by_name(node, field_name);

   if (
      field_value == NULL ||
      field_value->value_type != JSON_ARRAY ||
      field_value->array.size < length
   )
   {
      return 0;
   }

   for (unsigned int i = 0; i < length; ++i)
   {
      arr[i] = field_value->array.vals[i].inum;
   }

   return 1;
}

int copy_fixed_uint_array_field(
   const json_value_t * node,
   const char * field_name,
   unsigned int length,
   unsigned int * arr
)
{
   if (arr == NULL)
   {
      return 0;
   }

   const json_value_t * field_value = get_const_field_by_name(node, field_name);

   if (
      field_value == NULL ||
      field_value->value_type != JSON_ARRAY ||
      field_value->array.size < length
   )
   {
      return 0;
   }

   for (unsigned int i = 0; i < length; ++i)
   {
      arr[i] = (unsigned int )field_value->array.vals[i].inum;
   }

   return 1;
}

int copy_float_field(
   const json_value_t * node, const char * field_name, float * fnum
)
{
   if (fnum == NULL)
   {
      return 0;
   }
   const json_value_t * field_value = get_const_field_by_name(node, field_name);

   if (field_value == NULL || (field_value->value_type != JSON_FLOAT_NUMBER && field_value->value_type != JSON_INT_NUMBER))
   {
      return 0;
   }

   if (field_value->value_type == JSON_FLOAT_NUMBER)
   {
      *fnum = field_value->fnum;
   }
   else
   {
      *fnum = (float )field_value->inum;
   }

   return 1;
}

int copy_int_field(
   const json_value_t * node, const char * field_name, int * inum
)
{
   if (inum == NULL)
   {
      return 0;
   }
   const json_value_t * field_value = get_const_field_by_name(node, field_name);

   if (field_value == NULL || field_value->value_type != JSON_INT_NUMBER)
   {
      return 0;
   }

   *inum = field_value->inum;

   return 1;
}

int copy_uint_field(
   const json_value_t * node, const char * field_name, unsigned int * uinum
)
{
   if (uinum == NULL)
   {
      return 0;
   }
   const json_value_t * field_value = get_const_field_by_name(node, field_name);

   if (field_value == NULL || field_value->value_type != JSON_INT_NUMBER)
   {
      return 0;
   }

   *uinum = (unsigned int )field_value->inum;

   return 1;
}

int copy_string_field(
   const json_value_t * node,
   const char * field_name,
   char * out_str,
   unsigned int max_length
)
{
   if (node->value_type != JSON_OBJECT)
   {
      return 0;
   }

   for (unsigned int i = 0; i < max_length && i < node->string.size; ++i)
   {
      out_str[i] = node->string.buffer[i];
   }

   return 1;
}

int copy_object_field(
   const json_value_t * node,
   const char * field_name,
   data_from_json_f structifier,
   void * data
)
{
   const json_value_t * field_value = get_const_field_by_name(node, field_name);
   if (field_value == NULL || field_value->value_type != JSON_OBJECT)
   {
      return 0;
   }

   if (!structifier(field_value, data))
   {
      return 0;
   }

   return 1;
}

#ifdef __cplusplus
}
#endif
