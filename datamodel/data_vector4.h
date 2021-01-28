#ifndef DATA_VECTOR4_HEADER
#define DATA_VECTOR4_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct vector4
{
   float v[4];
} data_vector4_t;

void initialize_vector4(data_vector4_t * vec);

int vector4_to_json(json_value_t * node, const data_vector4_t * data);

int vector4_from_json(const json_value_t * node, data_vector4_t * data);

int anon_vector4_to_json(json_value_t * node, const void * anon_data);

int anon_vector4_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
