#ifndef DATA_SHAPECAPSULE_HEADER
#define DATA_SHAPECAPSULE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct shapeCapsule
{
   float radius;
   float height;
} data_shapeCapsule_t;

void initialize_shapeCapsule(data_shapeCapsule_t * data);

int shapeCapsule_to_json(json_value_t * node, const data_shapeCapsule_t * data);

int shapeCapsule_from_json(const json_value_t * node, data_shapeCapsule_t * data);

int anon_shapeCapsule_to_json(json_value_t * node, const void * anon_data);

int anon_shapeCapsule_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
