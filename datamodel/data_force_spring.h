#ifndef DATA_FORCES_PRING_HEADER
#define DATA_FORCES_PRING_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct forceSpring
{
   int parentId;
   int childId;
   float restLength;
   float springCoeff;
   data_vector3_t parentLinkPoint;
   data_vector3_t childLinkPoint;
} data_forceSpring_t;

void initialize_forceSpring(data_forceSpring_t * data);

int forceSpring_to_json(json_value_t * node, const data_forceSpring_t * data);

int forceSpring_from_json(const json_value_t * node, data_forceSpring_t * data);

int anon_forceSpring_to_json(json_value_t * node, const void * anon_data);

int anon_forceSpring_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
