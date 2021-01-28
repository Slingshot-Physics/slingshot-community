#ifndef DATA_SHAPECYLINDER_HEADER
#define DATA_SHAPECYLINDER_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct shapeCylinder
{
   float radius;
   float height;
} data_shapeCylinder_t;

void initialize_shapeCylinder(data_shapeCylinder_t * data);

int shapeCylinder_to_json(json_value_t * node, const data_shapeCylinder_t * data);

int shapeCylinder_from_json(const json_value_t * node, data_shapeCylinder_t * data);

int anon_shapeCylinder_to_json(json_value_t * node, const void * anon_data);

int anon_shapeCylinder_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
