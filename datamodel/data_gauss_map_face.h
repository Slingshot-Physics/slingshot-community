#ifndef DATA_GAUSS_MAP_FACE_HEADER
#define DATA_GAUSS_MAP_FACE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct data_gaussMapFace_s
{
   data_vector3_t normal;

   unsigned int triangleStartId;

   unsigned int numTriangles;
} data_gaussMapFace_t;

void initialize_gaussMapFace(data_gaussMapFace_t * data);

int gaussMapFace_to_json(json_value_t * node, const data_gaussMapFace_t * data);

int gaussMapFace_from_json(const json_value_t * node, data_gaussMapFace_t * data);

int anon_gaussMapFace_to_json(json_value_t * node, const void * anon_data);

int anon_gaussMapFace_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
