#ifndef DATA_MESHTRIANGLE_HEADER
#define DATA_MESHTRIANGLE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

// Triangles will reference a vertex by its index in a larger mesh.
typedef struct meshTriangle
{
   unsigned int vertIds[3];
   data_vector3_t normal;
} data_meshTriangle_t;

void initialize_meshTriangle(data_meshTriangle_t * triangle);

int meshTriangle_to_json(json_value_t * node, const data_meshTriangle_t * data);

int meshTriangle_from_json(const json_value_t * node, data_meshTriangle_t * data);

int anon_meshTriangle_to_json(json_value_t * node, const void * anon_data);

int anon_meshTriangle_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
