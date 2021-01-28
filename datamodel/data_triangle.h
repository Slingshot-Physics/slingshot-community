#ifndef DATA_TRIANGLE_HEADER
#define DATA_TRIANGLE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct data_triangle_s
{
   data_vector3_t verts[3];
} data_triangle_t;

void initialize_triangle(data_triangle_t * data);

int triangle_to_json(json_value_t * node, const data_triangle_t * data);

int triangle_from_json(const json_value_t * node, data_triangle_t * data);

int anon_triangle_to_json(json_value_t * node, const void * anon_data);

int anon_triangle_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
