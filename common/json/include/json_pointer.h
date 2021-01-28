#ifndef JSON_POINTER_HEADER
#define JSON_POINTER_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

void json_pointer_tokenize(const char * pointer_str, json_array_t * tokens); 

K_POINTERTOKENTYPE json_pointer_token_type(json_string_t * token_str);

#ifdef __cplusplus
}
#endif

#endif
