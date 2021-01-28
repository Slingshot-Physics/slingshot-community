#ifndef DATA_POLYGON50_HEADER
#define DATA_POLYGON50_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct polygon50
{
   unsigned int numVerts;
   data_vector3_t verts[50];
} data_polygon50_t;

void initialize_polygon50(data_polygon50_t * data);

int polygon50_to_json(json_value_t * node, const data_polygon50_t * data);

int polygon50_from_json(const json_value_t * node, data_polygon50_t * data);

int anon_polygon50_to_json(json_value_t * node, const void * anon_data);

int anon_polygon50_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
