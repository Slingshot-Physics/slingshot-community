#include "json_string.h"

#include <stdlib.h>
#include <string.h>

#define DELTA_CAPACITY 32
#define MAX_STRING_MAKE_SIZE 65535

void json_string_initialize(json_string_t * data)
{
   memset(data, 0, sizeof(json_string_t));
}

int json_string_make(json_string_t * buff, const char * in_string)
{
   int i = 0;
   char temp_char = 'a';
   while (
      (temp_char != '\0') &&
      (buff->size < MAX_STRING_MAKE_SIZE)
   )
   {
      temp_char = in_string[i];
      json_string_append(buff, temp_char);
      ++i;
   }

   return buff->size < MAX_STRING_MAKE_SIZE;
}

void json_string_delete(json_string_t * data)
{
   free(data->buffer);
   data->buffer = NULL;
   data->size = 0;
   data->capacity = 0;
}

void json_string_clear(json_string_t * buff)
{
   buff->size = 0;
}

void json_string_allocate(json_string_t * data)
{
   data->capacity = DELTA_CAPACITY;
   data->size = 0;
   data->buffer = malloc(data->capacity * sizeof(char));
}

void json_string_increase_capacity(json_string_t * data)
{
   char * temp_ptr = malloc(
      (data->capacity + DELTA_CAPACITY) * sizeof(char)
   );

   memcpy(temp_ptr, data->buffer, data->size * sizeof(char));

   free(data->buffer);

   data->buffer = temp_ptr;
   data->capacity += DELTA_CAPACITY;
}

void json_string_append(json_string_t * data, char element)
{
   if ((data->size + 1) >= data->capacity)
   {
      json_string_increase_capacity(data);
   }

   data->buffer[data->size] = element;
   data->size += 1;
}

int json_string_compare(const json_string_t * a, const json_string_t * b)
{
   if (a->size != b->size)
   {
      return 0;
   }

   for (unsigned int i = 0; i < a->size; ++i)
   {
      if (a->buffer[i] != b->buffer[i])
      {
         return 0;
      }
   }

   return 1;
}

void json_string_assign(json_string_t * dst, const json_string_t * src)
{
   if (src->size == 0)
   {
      return;
   }

   json_string_clear(dst);

   for (unsigned int i = 0; i < src->size; ++i)
   {
      json_string_append(dst, src->buffer[i]);
   }
}
