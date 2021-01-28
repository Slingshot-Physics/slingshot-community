#ifdef __cplusplus
extern "C"
{
#endif


#include "json_serialize.h"

#include "json_array.h"
#include "json_string.h"

#include <stdio.h>
#include <stdlib.h>

void json_serialize_string(
   const json_value_t * node, int level, json_string_t * out
)
{
   (void)level;
   json_string_append(out, '\"');

   for (unsigned int i = 0; i < node->string.size; ++i)
   {
      char temp_char = node->string.buffer[i];
      if (temp_char != '\0')
      {
         json_string_append(out, node->string.buffer[i]);
      }
   }

   json_string_append(out, '\"');
}

void json_serialize_object(
   const json_value_t * node, int level, json_string_t * out
)
{
   (void)level;
   json_string_append(out, '{');

   for (unsigned int i = 0; i < node->object.size; ++i)
   {
      json_serialize_string(&(node->object.keys[i]), level, out);
      json_string_append(out, ':');
      json_serialize_value(&(node->object.values[i]), level, out);
      if (i < node->object.size - 1)
      {
         json_string_append(out, ',');
      }
   }

   json_string_append(out, '}');
}

void json_serialize_array(
   const json_value_t * node, int level, json_string_t * out
)
{
   (void)level;
   json_string_append(out, '[');

   for (unsigned int i = 0; i < node->array.size; ++i)
   {
      json_serialize_value(&(node->array.vals[i]), level, out);
      if (i < node->object.size - 1)
      {
         json_string_append(out, ',');
      }
   }

   json_string_append(out, ']');
}

void json_serialize_number(const json_value_t * node, int level, json_string_t * out)
{
   (void)level;
   char temp_str[32];

   switch(node->value_type)
   {
      case JSON_INT_NUMBER:
      {
         snprintf(temp_str, 32, "%d", node->inum);
         break;
      }
      case JSON_FLOAT_NUMBER:
      {
         snprintf(temp_str, 32, "%0.12f", node->fnum);
         break;
      }
      default:
         break;
   }

   for (int i = 0; ; ++i)
   {
      char temp_char = temp_str[i];
      if (temp_char == '\0')
      {
         break;
      }
      json_string_append(out, temp_char);
   }
}

void json_serialize_value(
   const json_value_t * node, int level, json_string_t * out
)
{
   (void)level;
   switch(node->value_type)
   {
      case JSON_STRING:
      {
         json_serialize_string(node, level, out);
         break;
      }
      case JSON_FLOAT_NUMBER:
      case JSON_INT_NUMBER:
      {
         json_serialize_number(node, level, out);
         break;
      }
      case JSON_ARRAY:
      {
         json_serialize_array(node, level, out);
         break;
      }
      case JSON_OBJECT:
      {
         json_serialize_object(node, level, out);
         break;
      }
      default:
         break;
   }
}

void json_serialize_to_str(const json_value_t * value, json_string_t * out)
{
   json_string_initialize(out);
   json_string_allocate(out);

   int level = 0;

   if (
      (value->value_type != JSON_OBJECT) ||
      (value == NULL) ||
      (out == NULL)
   )
   {
      return;
   }

   json_serialize_object(value, level, out);

   json_string_append(out, '\0');
}

void json_serialize_to_file(const json_value_t * value, FILE * file_ptr)
{
   if (file_ptr == NULL)
   {
      return;
   }

   json_string_t temp_str;
   json_serialize_to_str(value, &temp_str);

   fprintf(file_ptr, "%s\n", temp_str.buffer);

   json_string_delete(&temp_str);
}

#ifdef __cplusplus
}
#endif
