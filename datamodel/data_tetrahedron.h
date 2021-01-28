#ifndef DATA_TETRAHEDRON_HEADER
#define DATA_TETRAHEDRON_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct data_tetrahedron_s
{
   data_vector3_t verts[4];
} data_tetrahedron_t;

void initialize_tetrahedron(data_tetrahedron_t * data);

int tetrahedron_to_json(json_value_t * node, const data_tetrahedron_t * data);

int tetrahedron_from_json(const json_value_t * node, data_tetrahedron_t * data);

int anon_tetrahedron_to_json(json_value_t * node, const void * anon_data);

int anon_tetrahedron_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
