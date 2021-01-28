#include "json_pointer.h"

#include "json_array.h"
#include "json_char_ops.h"
#include "json_string.h"

#include <stdlib.h>

#define MAX_JSON_POINTER_STR_LEN 65535

// Copies everything except '/' characters (solidus) to the token string.
static void convert_buffer_to_token(
   const json_string_t * const buffer, json_string_t * token
)
{
   for (unsigned int i = 0; i < buffer->size; ++i)
   {
      if (buffer->buffer[i] != '/')
      {
         json_string_append(token, buffer->buffer[i]);
      }
   }
}

K_POINTERTOKENTYPE json_pointer_token_type(
   json_string_t * token_str
)
{
   K_POINTERTOKENTYPE token_type = POINTER_TOKEN_INT;

   // Determine the token type by looking at it.
   for (unsigned int j = 0; j < token_str->size - 1; ++j)
   {
      if (
         (token_type == POINTER_TOKEN_INT) &&
         !( // There's an exclamation point here.
            (json_character_type(token_str->buffer[j]) == CHAR_ZERO) ||
            (json_character_type(token_str->buffer[j]) == CHAR_ONENINE)
         )
      )
      {
         token_type = POINTER_TOKEN_STR;
         break;
      }
   }

   return token_type;
}

void json_pointer_tokenize(const char * pointer_str, json_array_t * tokens)
{
   json_array_initialize(tokens);
   json_array_allocate(tokens);

   if (pointer_str == NULL || pointer_str[0] == '\0')
   {
      return;
   }

   json_string_t temp_token;
   json_string_initialize(&temp_token);
   json_string_allocate(&temp_token);

   json_string_t pointer_buffer;
   json_string_initialize(&pointer_buffer);
   json_string_allocate(&pointer_buffer);

   char temp_char;
   unsigned int i = 0;
   do
   {
      temp_char = pointer_str[i];

      if (temp_char != '/' && temp_char != '\0')
      {
         json_string_append(&pointer_buffer, temp_char);
      }

      if (
         (pointer_buffer.size > 0) &&
         (temp_char == '/' || temp_char == '\0')
      )
      {
         json_string_append(&pointer_buffer, '\0');

         convert_buffer_to_token(&pointer_buffer, &temp_token);
         json_value_t temp_value;
         temp_value.value_type = JSON_STRING;
         json_string_initialize(&temp_value.string);
         json_string_allocate(&temp_value.string);
         json_string_assign(&temp_value.string, &temp_token);

         json_array_append(tokens, &temp_value);

         json_string_clear(&temp_token);
         json_string_clear(&pointer_buffer);
      }

      ++i;
   } while (temp_char != '\0' && i < MAX_JSON_POINTER_STR_LEN);

   json_string_delete(&temp_token);
   json_string_delete(&pointer_buffer);
}
