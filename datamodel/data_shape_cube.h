#ifndef DATA_SHAPE_CUBE_HEADER
#define DATA_SHAPE_CUBE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct data_shapeCube_s
{
   // x-axis
   float length;

   // y-axis
   float width;

   // z-axis
   float height;
} data_shapeCube_t;

void initialize_shapeCube(data_shapeCube_t * data);

int shapeCube_to_json(json_value_t * node, const data_shapeCube_t * data);

int shapeCube_from_json(const json_value_t * node, data_shapeCube_t * data);

int anon_shapeCube_to_json(json_value_t * node, const void * anon_data);

int anon_shapeCube_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
