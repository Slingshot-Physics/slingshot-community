#ifdef __cplusplus
extern "C"
{
#endif


#include "json_object.h"

#include "json_string.h"

#include <stdlib.h>
#include <string.h>

#define DELTA_CAPACITY 32

void json_object_initialize(json_object_t * data)
{
   memset(data, 0, sizeof(json_object_t));
}

void json_object_delete(json_object_t * data)
{
   free(data->keys);
   data->keys = NULL;
   free(data->values);
   data->values = NULL;
   data->size = 0;
   data->capacity = 0;
}

void json_object_allocate(json_object_t * data)
{
   data->capacity = DELTA_CAPACITY;
   data->size = 0;
   data->keys = malloc(data->capacity * sizeof(json_value_t));
   data->values = malloc(data->capacity * sizeof(json_value_t));
}

void json_object_increase_capacity(json_object_t * data)
{
   json_value_t * temp_key_ptr = malloc(
      (data->capacity + DELTA_CAPACITY) * sizeof(json_value_t)
   );

   memcpy(temp_key_ptr, data->keys, data->size * sizeof(json_value_t));

   free(data->keys);

   data->keys = temp_key_ptr;

   json_value_t * temp_value_ptr = malloc(
      (data->capacity + DELTA_CAPACITY) * sizeof(json_value_t)
   );

   memcpy(temp_value_ptr, data->values, data->size * sizeof(json_value_t));

   free(data->values);

   data->values = temp_value_ptr;

   data->capacity += DELTA_CAPACITY;
}

void json_object_append(
   json_object_t * data, const json_value_t * key, const json_value_t * value
)
{
   if (
      (key->value_type != JSON_STRING) ||
      (key->string.size == 0)
   )
   {
      return;
   }

   if ((data->size + 1) >= data->capacity)
   {
      json_object_increase_capacity(data);
   }

   data->keys[data->size] = *key;

   data->values[data->size] = *value;

   data->size += 1;
}

void json_object_append_key(json_object_t * data, const json_value_t * key)
{
   if ((key->value_type != JSON_STRING))
   {
      return;
   }

   if ((data->size + 1) >= data->capacity)
   {
      json_object_increase_capacity(data);
   }

   data->keys[data->size] = *key;

   data->values[data->size].value_type = JSON_NONE;

   data->size += 1;
}

void json_object_append_value(json_object_t * data, const json_value_t * value)
{
   data->values[data->size - 1] = *value;
}

json_value_t * json_object_find_by_key(
   const json_object_t * data, const json_string_t * key
)
{
   json_value_t * result = NULL;

   // Keys have to be non-zero length strings.
   if (key->size == 0)
   {
      return result;
   }

   for (unsigned int i = 0; i < data->size; ++i)
   {
      if (json_string_compare(&(data->keys[i].string), key))
      {
         result = &(data->values[i]);
         break;
      }
   }

   return result;
}

#ifdef __cplusplus
}
#endif
