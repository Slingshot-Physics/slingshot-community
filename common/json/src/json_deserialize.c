#ifdef __cplusplus
extern "C"
{
#endif

#include "json_deserialize.h"

#include "json_array.h"
#include "json_char_ops.h"
#include "json_object.h"
#include "json_string.h"
#include "json_value.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void json_strip(
   const json_string_t * in_json, json_string_t * out_json
)
{
   int escape_code = 0;
   int inside_string = 0;
   json_string_initialize(out_json);
   json_string_allocate(out_json);

   for (unsigned int i = 0; i < in_json->size; ++i)
   {
      K_CHARACTERTYPE char_type = json_character_type(in_json->buffer[i]);

      if (
         (char_type == CHAR_REVERSE_SOLIDUS) ||
         escape_code
      )
      {
         ++escape_code;
      }

      if (char_type == CHAR_QUOTATION_MARK && !escape_code)
      {
         inside_string = 1 - inside_string;
      }

      if (escape_code == 2)
      {
         escape_code = 0;
      }

      if (
         (char_type == CHAR_WHITESPACE) &&
         !inside_string &&
         !escape_code
      )
      {
         continue;
      }

      json_string_append(out_json, in_json->buffer[i]);
   }

   json_string_append(out_json, '\0');
}

void json_merge_buffer_to_value(
   const json_string_t * buffer, json_value_t * val
)
{
   switch(val->value_type)
   {
      case JSON_INT_NUMBER:
      {
         val->inum = atoi(buffer->buffer);
         break;
      }
      case JSON_FLOAT_NUMBER:
      {
         val->fnum = atof(buffer->buffer);
         break;
      }
      case JSON_STRING:
      {
         for (unsigned int i = 0; i < buffer->size; ++i)
         {
            json_string_append(&val->string, buffer->buffer[i]);
         }
         break;
      }
      default:
      {
         printf("ooohhh you tried to merge a buffer to an undetermined value\n");
         break;
      }
   }
}

parse_f json_assign_parser(const json_value_t * state)
{
   if (state == NULL)
   {
      return NULL;
   }

   switch(state->value_type)
   {
      case JSON_STRING:
         return json_parse_string;
      case JSON_NONE:
         return json_parse_value;
      case JSON_INT_NUMBER:
         return json_parse_number;
      case JSON_FLOAT_NUMBER:
         return json_parse_fraction;
      case JSON_ARRAY:
         return json_parse_array;
      case JSON_OBJECT:
         return json_parse_object;
      default:
         break;
   }

   return json_parse_value;
}

void json_parse_value(json_parse_state_t * state, char c)
{
   switch(json_character_type(c))
   {
      case CHAR_QUOTATION_MARK:
      {
         json_string_clear(&state->buffer);
         json_value_allocate_type(state->node, JSON_STRING);
         break;
      }
      case CHAR_ONENINE:
      case CHAR_MINUS:
      case CHAR_ZERO:
      {
         json_string_clear(&state->buffer);
         json_string_append(&state->buffer, c);
         state->node->value_type = JSON_INT_NUMBER;
         break;
      }
      case CHAR_OPEN_CURLY_BRACKET:
      {
         json_value_allocate_type(state->node, JSON_OBJECT);
         break;
      }
      case CHAR_OPEN_SQUARE_BRACKET:
      {
         json_value_allocate_type(state->node, JSON_ARRAY);
         state->node = json_value_append_new_blank_value_to_array(state->node);
         break;
      }
      case CHAR_CLOSE_SQUARE_BRACKET:
      {
         // For cases where we encounter empty arrays
         state->node = state->node->parent;
         // If the array is empty in the JSON string, then it will still have
         // a non-zero size (see the open square bracket case for value
         // parsing). So before returning to the grandparent node we check to
         // see if the array has size one and an array element of none type.
         // If this is the case, then we set the array's size to zero.
         if (state->node->value_type == JSON_ARRAY)
         {
            if (
               (state->node->array.size == 1) &&
               (state->node->array.vals[0].value_type == JSON_NONE)
            )
            {
               state->node->array.size = 0;
            }
         }
         state->node = state->node->parent;
         break;
      }
      case CHAR_CLOSE_CURLY_BRACKET:
      {
         // For cases where we encounter empty objects
         state->node = state->node->parent;
         state->node = state->node->parent;
         break;
      }
      default:
         break;
   }

   state->parser = json_assign_parser(state->node);
}

void json_parse_string(json_parse_state_t * state, char c)
{
   switch(json_character_type(c))
   {
      case CHAR_QUOTATION_MARK:
      {
         json_string_append(&state->buffer, '\0');
         json_merge_buffer_to_value(&(state->buffer), state->node);
         json_string_clear(&state->buffer);
         state->node = state->node->parent;
         break;
      }
      default:
      {
         json_string_append(&state->buffer, c);
         break;
      }
   }

   state->parser = json_assign_parser(state->node);
}

void json_parse_number(json_parse_state_t * state, char c)
{
   switch(json_character_type(c))
   {
      case CHAR_PERIOD:
      {
         json_string_append(&state->buffer, c);
         state->node->value_type = JSON_FLOAT_NUMBER;
         break;
      }
      case CHAR_E:
      {
         json_string_append(&state->buffer, c);
         state->node->value_type = JSON_FLOAT_NUMBER;
         break;
      }
      case CHAR_COMMA:
      {
         json_string_append(&state->buffer, '\0');
         json_merge_buffer_to_value(&(state->buffer), state->node);
         json_string_clear(&state->buffer);
         // There will be more things added to the parent node.
         state->node = state->node->parent;

         // Numbers only terminate on commas and brackets. Terminating on a
         // comma means that another value will be added to the current
         // container.
         // If the parent container for this number is an object, then calling
         // the object_parser on the comma character has no effect.
         // If the parent container for this number is an array, then calling
         // the array_parser on the comma character has the effect of
         // allocating a new value for the array and setting the parser to the
         // value_parser.
         state->parser = json_assign_parser(state->node);
         state->parser(state, c);
         break;
      }
      case CHAR_CLOSE_CURLY_BRACKET:
      case CHAR_CLOSE_SQUARE_BRACKET:
      {
         json_string_append(&state->buffer, '\0');
         json_merge_buffer_to_value(&(state->buffer), state->node);
         json_string_clear(&state->buffer);
         // I need to escape this node AND the parent node because the parent
         // node has been terminated in addition to this node.
         state->node = state->node->parent;
         state->node = state->node->parent;
         break;
      }
      default:
      {
         json_string_append(&state->buffer, c);
         break;
      }
   }

   state->parser = json_assign_parser(state->node);
}

void json_parse_fraction(json_parse_state_t * state, char c)
{
   switch(json_character_type(c))
   {
      case CHAR_COMMA:
      {
         json_string_append(&state->buffer, '\0');
         json_merge_buffer_to_value(&(state->buffer), state->node);
         json_string_clear(&state->buffer);
         // There will be more things added to the parent node.
         state->node = state->node->parent;

         // This is a little weird, but maybe it'll work.
         state->parser = json_assign_parser(state->node);
         state->parser(state, c);
         break;
      }
      case CHAR_CLOSE_CURLY_BRACKET:
      case CHAR_CLOSE_SQUARE_BRACKET:
      {
         json_string_append(&state->buffer, '\0');
         json_merge_buffer_to_value(&(state->buffer), state->node);
         json_string_clear(&state->buffer);
         // I need to escape this node AND the parent node because the parent
         // node has been terminated in addition to this node.
         state->node = state->node->parent;
         state->node = state->node->parent;
         break;
      }
      default:
      {
         json_string_append(&state->buffer, c);
         break;
      }
   }

   state->parser = json_assign_parser(state->node);
}

void json_parse_exponent(json_parse_state_t * state, char c)
{
   switch(json_character_type(c))
   {
      case CHAR_COMMA:
      {
         json_string_append(&state->buffer, '\0');
         json_merge_buffer_to_value(&(state->buffer), state->node);
         json_string_clear(&state->buffer);
         // There will be more things added to the parent node.
         state->node = state->node->parent;

         // This is a little weird, but maybe it'll work.
         state->parser = json_assign_parser(state->node);
         state->parser(state, c);
         break;
      }
      case CHAR_CLOSE_CURLY_BRACKET:
      case CHAR_CLOSE_SQUARE_BRACKET:
      {
         json_string_append(&state->buffer, '\0');
         json_merge_buffer_to_value(&(state->buffer), state->node);
         json_string_clear(&state->buffer);
         // I need to escape this node AND the parent node because the parent
         // node has been terminated in addition to this node.
         state->node = state->node->parent;
         state->node = state->node->parent;
         break;
      }
      default:
      {
         json_string_append(&state->buffer, c);
         break;
      }
   }

   state->parser = json_assign_parser(state->node);
}

void json_parse_object(json_parse_state_t * state, char c)
{
   switch(json_character_type(c))
   {
      case CHAR_COMMA:
      {
         break;
      }
      case CHAR_QUOTATION_MARK:
      {
         state->node = json_value_append_new_blank_key_to_object(state->node);
         json_string_initialize(&(state->node->string));
         json_string_allocate(&(state->node->string));
         break;
      }
      case CHAR_COLON:
      {
         state->node = json_value_append_new_blank_value_to_object(state->node);
         break;
      }
      case CHAR_CLOSE_CURLY_BRACKET:
      {
         state->node = state->node->parent;
         break;
      }
      default:
         break;
   }

   state->parser = json_assign_parser(state->node);
}

void json_parse_array(json_parse_state_t * state, char c)
{
   switch(json_character_type(c))
   {
      case CHAR_COMMA:
      {
         // Is this correct?
         state->node = json_value_append_new_blank_value_to_array(state->node);
         break;
      }
      case CHAR_CLOSE_SQUARE_BRACKET:
      {
         // This might be dead code.
         state->node = state->node->parent;
         break;
      }
      default:
         break;
   }

   state->parser = json_assign_parser(state->node);
}

static int json_deserialize_stripped_str(
   json_string_t * stripped_json_str, json_value_t * root
)
{
   json_value_t * data = root;
   json_value_initialize(data);
   data->value_type = JSON_NONE;

   json_parse_state_t state;
   state.node = root;
   state.node->parent = NULL;
   json_string_initialize(&state.buffer);
   json_string_allocate(&state.buffer);
   state.parser = json_assign_parser(state.node);

   for (unsigned int i = 0; i < stripped_json_str->size; ++i)
   {
      char temp_char = stripped_json_str->buffer[i];
      if (state.parser != NULL)
      {
         state.parser(&state, temp_char);
      }
   }

   json_string_delete(&state.buffer);

   return 1;
}

int json_deserialize_str(json_string_t * json_data, json_value_t * root)
{
   json_string_t stripped_json_str;

   json_strip(json_data, &stripped_json_str);

   int result = json_deserialize_stripped_str(&stripped_json_str, root);
   json_string_delete(&stripped_json_str);
   return result;
}

int json_deserialize_file(FILE * file_ptr, json_value_t * root)
{
   int file_char = 0;
   json_string_t full_json_str;
   json_string_initialize(&full_json_str);
   json_string_allocate(&full_json_str);
   while (file_char != EOF)
   {
      file_char = fgetc(file_ptr);
      if (file_char > 0)
      {
         json_string_append(&full_json_str, file_char);
      }
   }

   json_string_t stripped_json_str;
   // Don't need to allocate the stripped string because json_strip will do
   // that for us.
   json_strip(&full_json_str, &stripped_json_str);

   int result = json_deserialize_stripped_str(&stripped_json_str, root);
   json_string_delete(&stripped_json_str);
   json_string_delete(&full_json_str);

   return result;
}

#ifdef __cplusplus
}
#endif
