#ifdef __cplusplus
extern "C"
{
#endif

#include "json_verify.h"

#include "json_char_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DELTA_CAPACITY 32

#define BAD_TRANSITION -9
#define GO_TO_PARENT_CONTAINER -1
#define GO_TO_GRANDPARENT_CONTAINER -2
#define GO_TO_PARENT_TYPE -5

static int transitions[16][24] = {

//                        ws,  ", 1-9, an,  :,  ,,  {,  },  [,  ],  0,  -,  ., eE,  +,  \, tT, rR, uU, fF, aA, lL, sS, nN
/* object          */   {  0,  1,  -9, -9,  2, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* object_key      */   {  1,  6,  -9,  6, -9, -9, -9, -9, -9, -9, -9, -9, -9,  6, -9, -9,  6,  6,  6,  6,  6,  6,  6,  6 },
/* object_value    */   {  2,  6,   3, -9, -9,  1,  0, -9, 15, -9,  3,  3, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* number          */   { -9, -9,   3, -9, -9, -1, -9, -2, -9, -2,  3,  3,  4,  5, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* fraction        */   { -9, -9,   4, -9, -9, -1, -9, -2, -9, -2,  4, -9, -9,  5, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* exponent        */   { -9, -9,   5, -9, -9, -1, -9, -2, -9, -2,  5,  5, -9, -9,  5, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* string          */   {  6, -1,   6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  8,  6,  6,  6,  6,  6,  6,  6,  6 },
/* value           */   {  7,  6,   3, -9, -9, -9,  0, -9, 15, -9,  3,  3, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* escape          */   { -9,  9,  -9,  9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9,  9,  9,  9, 10,  9, -9, -9, -9,  9 },
/* escape1         */   { -5, -1,  -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5 },
/* escape_unicode  */   { -9, -9,  11, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* escape_unicode1 */   { -9, -9,  12, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* escape_unicode2 */   { -9, -9,  13, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* escape_unicode3 */   { -9, -9,  14, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 },
/* escape_unicode4 */   { -5, -1,  -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5 },
/* array           */   { 15,  6,   3, -9, -9,  7,  0, -9, 15, -1,  3,  3, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9 }
};

void json_parse_stack_initialize(json_parse_stack_t * data)
{
   memset(data, 0, sizeof(json_parse_stack_t));
}

void json_parse_stack_delete(json_parse_stack_t * data)
{
   free(data->vals);
   data->vals = NULL;
   data->size = 0;
   data->capacity = 0;
}

void json_parse_stack_allocate(json_parse_stack_t * data)
{
   data->capacity = DELTA_CAPACITY;
   data->size = 0;
   data->vals = (K_PARSESTATE * )malloc(
      data->capacity * sizeof(K_PARSESTATE)
   );
}

void json_parse_stack_increase_capacity(json_parse_stack_t * data)
{
   K_PARSESTATE * temp_ptr = (K_PARSESTATE * )malloc(
      (data->capacity + DELTA_CAPACITY) * sizeof(K_PARSESTATE)
   );

   memcpy(temp_ptr, data->vals, data->size * sizeof(K_PARSESTATE));

   free(data->vals);

   data->vals = temp_ptr;
   data->capacity += DELTA_CAPACITY;
}

void json_parse_stack_append(json_parse_stack_t * data, K_PARSESTATE state)
{
   if ((data->size + 1) >= data->capacity)
   {
      json_parse_stack_increase_capacity(data);
   }

   data->vals[data->size] = state;
   data->size += 1;
}

K_PARSESTATE json_parse_stack_pop_last(json_parse_stack_t * data)
{
   K_PARSESTATE last_val = (data->vals[data->size - 1]);
   data->size -= 1;
   data->size = data->size < 0 ? 0 : data->size;
   return last_val;
}

void json_print_parse_state(K_PARSESTATE parse_state)
{
   switch(parse_state)
   {
      case PARSE_OBJECT:
         printf("object\n");
         break;
      case PARSE_OBJECT_KEY:
         printf("object key\n");
         break;
      case PARSE_OBJECT_VALUE:
         printf("object value\n");
         break;
      case PARSE_NUMBER:
         printf("number\n");
         break;
      case PARSE_FRACTION:
         printf("fraction\n");
         break;
      case PARSE_EXPONENT:
         printf("exponent\n");
         break;
      case PARSE_STRING:
         printf("string\n");
         break;
      case PARSE_VALUE:
         printf("value\n");
         break;
      case PARSE_ESCAPE:
         printf("escape\n");
         break;
      case PARSE_ESCAPE1:
         printf("escape1\n");
         break;
      case PARSE_ESCAPE_UNICODE:
         printf("unicode\n");
         break;
      case PARSE_ESCAPE_UNICODE1:
         printf("unicode1\n");
         break;
      case PARSE_ESCAPE_UNICODE2:
         printf("unicode2\n");
         break;
      case PARSE_ESCAPE_UNICODE3:
         printf("unicode3\n");
         break;
      case PARSE_ESCAPE_UNICODE4:
         printf("unicode4\n");
         break;
      case PARSE_ARRAY:
         printf("array\n");
         break;
      default:
         printf("unknown\n");
         break;
   }
}

void json_verify_print_stack(json_parse_stack_t * stack)
{
   for (int i = 0; i < stack->size; ++i)
   {
      json_print_parse_state(stack->vals[i]);
   }
}

int json_verify_state_termination(
   K_PARSESTATE parse_state, K_CHARACTERTYPE char_type
)
{
   if (parse_state == PARSE_OBJECT && char_type == CHAR_CLOSE_CURLY_BRACKET)
   {
      return 1;
   }

   if (parse_state == PARSE_ARRAY && char_type == CHAR_CLOSE_SQUARE_BRACKET)
   {
      return 1;
   }

   return -1;
}

int json_verify_goto_parent_container(json_parse_stack_t * stack)
{
   K_PARSESTATE current_state = json_parse_stack_pop_last(stack);
   while (
      (stack->size != 0) &&
      (current_state != PARSE_OBJECT) &&
      (current_state != PARSE_ARRAY)
   )
   {
      current_state = json_parse_stack_pop_last(stack);
   }

   // There is no parent, but there should be.
   if (stack->size == 0)
   {
      return -1;
   }

   return 1;
}

int json_verify_goto_parent_type(json_parse_stack_t * stack)
{
   K_PARSESTATE current_state = json_parse_stack_pop_last(stack);
   while (
      (stack->size != 0) &&
      (current_state != PARSE_NUMBER) &&
      (current_state != PARSE_STRING)
   )
   {
      current_state = json_parse_stack_pop_last(stack);
   }

   // There is no parent, but there should be.
   if (stack->size == 0)
   {
      return -1;
   }

   return 1;
}

int json_verify_char_type(
   json_parse_stack_t * stack, K_CHARACTERTYPE char_type
)
{
   K_PARSESTATE curr_state = stack->vals[stack->size - 1];
   int new_state_cmd = transitions[(int )curr_state][(int )char_type];

   if (new_state_cmd == BAD_TRANSITION)
   {
      printf("bad transition\n");
      return -1;
   }

   switch(new_state_cmd)
   {
      case GO_TO_PARENT_TYPE:
      {
         if (json_verify_goto_parent_type(stack) < 0)
         {
            return -1;
         }
         break;
      }
      case GO_TO_PARENT_CONTAINER:
      {
         if (
            (json_verify_goto_parent_container(stack) < 0) ||
            (json_verify_state_termination(curr_state, char_type) < 0)
         )
         {
            return -1;
         }
         
         break;
      }
      case GO_TO_GRANDPARENT_CONTAINER:
      {
         if (
            (json_verify_goto_parent_container(stack) < 0) ||
            (json_verify_state_termination(curr_state, char_type) < 0) ||
            (json_verify_goto_parent_container(stack) < 0) ||
            (json_verify_state_termination(curr_state, char_type) < 0)
         )
         {
            return -1;
         }

         break;
      }
      default:
      {
         if (curr_state == PARSE_VALUE)
         {
            json_parse_stack_pop_last(stack);
         }
         json_parse_stack_append(stack, new_state_cmd);
         break;
      }
   }

   return 1;
}

int json_verify_basic(json_string_t * json_data)
{
   int escape_code = 0;
   int inside_string = 0;

   if (json_data == NULL)
   {
      return 0;
   }

   int num_curly_braces = 0;
   int num_square_braces = 0;
   int num_quotation_marks = 0;

   for (unsigned int i = 0; i < json_data->size; ++i)
   {
      K_CHARACTERTYPE char_type = json_character_type(json_data->buffer[i]);

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

      if ((!inside_string && !escape_code))
      {
         num_curly_braces += (char_type == CHAR_OPEN_CURLY_BRACKET) + (char_type == CHAR_CLOSE_CURLY_BRACKET);
         num_square_braces += (char_type == CHAR_OPEN_SQUARE_BRACKET) + (char_type == CHAR_CLOSE_SQUARE_BRACKET);
      }

      if (!escape_code)
      {
         num_quotation_marks += (char_type == CHAR_QUOTATION_MARK);
      }
   }

   printf("num curly braces: %d\n", num_curly_braces);
   printf("num square braces: %d\n", num_square_braces);
   printf("num quotation marks: %d\n", num_quotation_marks);

   return 2 * (
      (num_curly_braces % 2 == 0) &&
      (num_square_braces % 2 == 0) &&
      (num_quotation_marks % 2 == 0)
   ) - 1;
}

int json_verify(json_string_t * json_data)
{
   json_parse_stack_t parse_stack;
   json_parse_stack_initialize(&parse_stack);
   json_parse_stack_allocate(&parse_stack);

   if (json_data->buffer[0] != '{')
   {
      return -1;
   }
   else
   {
      json_parse_stack_append(&parse_stack, PARSE_OBJECT);
   }

   int result = 0;
   for (unsigned int i = 1; i < json_data->size; ++i)
   {
      char curr_char = json_data->buffer[i];
      result = json_verify_char_type(&parse_stack, json_character_type(curr_char));
      if (result < 0)
      {
         printf("json looks bad at character number %d with val %c\n", i, curr_char);
         json_verify_print_stack(&parse_stack);
         json_parse_stack_delete(&parse_stack);
         return -1;
      }
      json_verify_print_stack(&parse_stack);
   }
   printf("finished parsing\n");

   if (parse_stack.size == 0)
   {
      json_verify_print_stack(&parse_stack);
      json_parse_stack_delete(&parse_stack);
      return -1;
   }

   json_verify_print_stack(&parse_stack);
   json_parse_stack_delete(&parse_stack);
   return 1;
}

#ifdef __cplusplus
}
#endif
