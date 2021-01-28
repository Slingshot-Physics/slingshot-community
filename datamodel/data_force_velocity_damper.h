#ifndef DATA_FORCE_VELOCITY_DAMPER_HEADER
#define DATA_FORCE_VELOCITY_DAMPER_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct forceVelocityDamper
{
   int parentId;
   int childId;
   float damperCoeff;
   data_vector3_t parentLinkPoint;
   data_vector3_t childLinkPoint;
} data_forceVelocityDamper_t;

void initialize_forceVelocityDamper(data_forceVelocityDamper_t * data);

int forceVelocityDamper_to_json(json_value_t * node, const data_forceVelocityDamper_t * data);

int forceVelocityDamper_from_json(const json_value_t * node, data_forceVelocityDamper_t * data);

int anon_forceVelocityDamper_to_json(json_value_t * node, const void * anon_data);

int anon_forceVelocityDamper_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
