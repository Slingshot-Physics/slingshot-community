#ifndef JSON_STRING_HEADER
#define JSON_STRING_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

void json_string_initialize(json_string_t * buff);

// Sequentially adds the characters of 'in_string' into 'buff'. Returns 0 if
// 'in_string' is longer than the maximum string length, 1 otherwise.
int json_string_make(json_string_t * buff, const char * in_string);

void json_string_delete(json_string_t * buff);

void json_string_clear(json_string_t * buff);

void json_string_allocate(json_string_t * data);

void json_string_increase_capacity(json_string_t * data);

void json_string_append(json_string_t * data, char element);

// Returns 1 if they're equal, returns 0 if they're unequal.
int json_string_compare(const json_string_t * a, const json_string_t * b);

void json_string_assign(json_string_t * dst, const json_string_t * src);

#ifdef __cplusplus
}
#endif

#endif
