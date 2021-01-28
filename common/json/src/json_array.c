#ifdef __cplusplus
extern "C"
{
#endif


#include "json_array.h"

#include <stdlib.h>
#include <string.h>

#define DELTA_CAPACITY 32

void json_array_initialize(json_array_t * data)
{
   memset(data, 0, sizeof(json_array_t));
}

void json_array_delete(json_array_t * data)
{
   free(data->vals);
   data->vals = NULL;
   data->size = 0;
   data->capacity = 0;
}

void json_array_allocate(json_array_t * data)
{
   data->capacity = DELTA_CAPACITY;
   data->size = 0;
   data->vals = malloc(data->capacity * sizeof(json_value_t));
}

void json_array_increase_capacity(json_array_t * data)
{
   json_value_t * temp_ptr = malloc(
      (data->capacity + DELTA_CAPACITY) * sizeof(json_value_t)
   );

   memcpy(temp_ptr, data->vals, data->size * sizeof(json_value_t));

   free(data->vals);

   data->vals = temp_ptr;
   data->capacity += DELTA_CAPACITY;
}

void json_array_append(
   json_array_t * data, const json_value_t * element
)
{
   if ((data->size + 1) >= data->capacity)
   {
      json_array_increase_capacity(data);
   }

   data->vals[data->size] = *element;
   data->size += 1;
}

json_value_t json_array_pop_last(json_array_t * data)
{
   json_value_t last_val = (data->vals[data->size - 1]);
   data->size -= 1;
   return last_val;
}

#ifdef __cplusplus
}
#endif
