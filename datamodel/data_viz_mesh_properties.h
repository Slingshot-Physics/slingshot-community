#ifndef DATA_VIZMESHPROPERTIES_HEADER
#define DATA_VIZMESHPROPERTIES_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector4.h"

typedef struct vizMeshProperties
{
   unsigned int bodyId;

   data_vector4_t color;

} data_vizMeshProperties_t;

void initialize_vizMeshProperties(data_vizMeshProperties_t * data);

int vizMeshProperties_to_json(
   json_value_t * node, const data_vizMeshProperties_t * data
);

int vizMeshProperties_from_json(
   const json_value_t * node, data_vizMeshProperties_t * data
);

int anon_vizMeshProperties_to_json(json_value_t * node, const void * anon_data);

int anon_vizMeshProperties_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
