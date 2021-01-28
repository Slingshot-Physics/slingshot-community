#ifdef __cplusplus
extern "C"
{
#endif

#include "json_value.h"

#include "json_array.h"
#include "json_char_ops.h"
#include "json_object.h"
#include "json_pointer.h"
#include "json_string.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_JSON_POINTER_LEN 65535

void json_value_initialize(json_value_t * data)
{
   data->parent = NULL;
   data->value_type = JSON_NONE;
   data->inum = 0;
}

void json_value_allocate_type(
   json_value_t * value, K_VALUETYPE new_value_type
)
{
   value->value_type = new_value_type;

   switch(new_value_type)
   {
      case JSON_OBJECT:
      {
         json_object_initialize(&(value->object));
         json_object_allocate(&(value->object));
         for (unsigned int i = 0; i < value->object.capacity; ++i)
         {
            value->object.keys[i].value_type = JSON_STRING;
            value->object.keys[i].parent = value;
            value->object.values[i].value_type = JSON_NONE;
            value->object.values[i].parent = value;
         }
         break;
      }
      case JSON_ARRAY:
      {
         json_array_initialize(&(value->array));
         json_array_allocate(&(value->array));
         for (unsigned int i = 0; i < value->object.capacity; ++i)
         {
            value->array.vals[i].value_type = JSON_NONE;
            value->array.vals[i].parent = value;
         }
         break;
      }
      case JSON_STRING:
      {
         json_string_initialize(&(value->string));
         json_string_allocate(&(value->string));
         break;
      }
      default:
         break;
   }
}

json_value_t * json_value_append_new_blank_value_to_array(
   json_value_t * parent
)
{
   json_value_t new_value;
   new_value.parent = parent;
   new_value.value_type = JSON_NONE;
   json_array_append(&(parent->array), &new_value);
   return &(parent->array.vals[parent->array.size - 1]);
}

void json_value_append_value_to_array(
   json_value_t * parent, json_value_t * new_val
)
{
   new_val->parent = parent;
   json_array_append(&(parent->array), new_val);
}

json_value_t * json_value_append_new_blank_key_to_object(
   json_value_t * parent
)
{
   json_value_t new_key;
   new_key.parent = parent;
   new_key.value_type = JSON_STRING;
   json_object_append_key(&(parent->object), &new_key);
   return &(parent->object.keys[parent->object.size - 1]);
}

json_value_t * json_value_append_new_blank_value_to_object(
   json_value_t * parent
)
{
   json_value_t new_value;
   new_value.parent = parent;
   new_value.value_type = JSON_NONE;
   json_object_append_value(&(parent->object), &new_value);
   return &(parent->object.values[parent->object.size - 1]);
}

void json_value_delete_object(json_value_t * node)
{
   // Delete the keys first, they're just strings so it's no problem.
   for (unsigned int i = 0; i < node->object.size; ++i)
   {
      json_string_delete(&(node->object.keys[i].string));
   }

   // Delete any nested structures in the object's values.
   for (unsigned int i = 0; i < node->object.size; ++i)
   {
      json_value_delete(&(node->object.values[i]));
   }

   // Delete the allocations of json_value_t types as keys and values in the
   // object definition.
   json_object_delete(&(node->object));
}

void json_value_delete_array(json_value_t * node)
{
   // Delete any nested structures in the array's values.
   for (unsigned int i = 0; i < node->array.size; ++i)
   {
      json_value_delete(&(node->array.vals[i]));
   }

   // Delete the allocations of json_value_t types for the items in the array.
   json_array_delete(&(node->array));
}

void json_value_delete(json_value_t * node)
{
   switch(node->value_type)
   {
      case JSON_STRING:
      {
         json_string_delete(&(node->string));
         break;
      }
      case JSON_ARRAY:
      {
         json_value_delete_array(node);
         break;
      }
      case JSON_OBJECT:
      {
         json_value_delete_object(node);
         break;
      }
      default:
         break;
   }
}

int json_value_json_pointer_access(
   json_value_t * in_node,
   const char * json_pointer_str,
   json_value_t ** out_node
)
{
   json_value_t * cur_node = in_node;

   if (json_pointer_str == NULL || json_pointer_str[0] == '\0')
   {
      *out_node = in_node;
      return 1;
   }

   int success = 1;

   json_array_t tokens;

   json_pointer_tokenize(json_pointer_str, &tokens);

   for (unsigned int i = 0; i < tokens.size; ++i)
   {
      if (
         !(cur_node->value_type == JSON_ARRAY || cur_node->value_type == JSON_OBJECT)
      )
      {
         success = -1;
         cur_node = NULL;
         *out_node = cur_node;
         break;
      }

      K_POINTERTOKENTYPE token_type = json_pointer_token_type(&(tokens.vals[i].string));

      if (cur_node->value_type == JSON_ARRAY && token_type == POINTER_TOKEN_STR)
      {
         success = -1;
         cur_node = NULL;
         *out_node = cur_node;
         break;
      }
      else if (cur_node->value_type == JSON_ARRAY && token_type == POINTER_TOKEN_INT)
      {
         int int_token = atoi(tokens.vals[i].string.buffer);
         if (
            (cur_node->array.size == 0) ||
            ((unsigned int)int_token > cur_node->array.size - 1))
         {
            success = -1;
            cur_node = NULL;
            *out_node = cur_node;
            break;
         }

         cur_node = &(cur_node->array.vals[int_token]);
      }
      else if (cur_node->value_type == JSON_OBJECT)
      {
         cur_node = json_object_find_by_key(
            &(cur_node->object), &(tokens.vals[i].string)
         );

         if (cur_node == NULL)
         {
            success = -1;
            cur_node = NULL;
            *out_node = cur_node;
            break;
         }
      }
   }

   for (unsigned int i = 0; i < tokens.size; ++i)
   {
      json_string_delete(&(tokens.vals[i].string));
   }
   json_array_delete(&tokens);

   *out_node = cur_node;

   return success;
}

#ifdef __cplusplus
}
#endif
