#ifndef DATA_VECTOR3_HEADER
#define DATA_VECTOR3_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct vector3
{
   float v[3];
} data_vector3_t;

void initialize_vector3(data_vector3_t * vec);

int vector3_to_json(json_value_t * node, const data_vector3_t * data);

int vector3_from_json(const json_value_t * node, data_vector3_t * data);

int anon_vector3_to_json(json_value_t * node, const void * anon_data);

int anon_vector3_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
