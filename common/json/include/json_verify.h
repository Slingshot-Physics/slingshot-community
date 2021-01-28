#ifndef JSON_VERIFY_HEADER
#define JSON_VERIFY_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

void json_parse_stack_initialize(json_parse_stack_t * data);

void json_parse_stack_delete(json_parse_stack_t * data);

void json_parse_stack_allocate(json_parse_stack_t * data);

void json_parse_stack_increase_capacity(json_parse_stack_t * data);

void json_parse_stack_append(json_parse_stack_t * data, K_PARSESTATE state);

K_PARSESTATE json_parse_stack_pop_last(json_parse_stack_t * data);

void json_print_parse_state(K_PARSESTATE parse_state);

void json_verify_print_stack(json_parse_stack_t * stack);

int json_verify_state_termination(K_PARSESTATE parse_state, K_CHARACTERTYPE char_type);

int json_verify_goto_parent_container(json_parse_stack_t * stack);

int json_verify_goto_parent_type(json_parse_stack_t * stack);

int json_verify_char_type(json_parse_stack_t * stack, K_CHARACTERTYPE char_type);

// Returns 1 if the JSON looks good, -1 if it's bad, 0 if the data is NULL.
int json_verify_basic(json_string_t * json_data);

int json_verify(json_string_t * json_data);

#ifdef __cplusplus
}
#endif

#endif
