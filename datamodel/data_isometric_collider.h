#ifndef DATA_ISOMETRIC_COLLIDER_HEADER
#define DATA_ISOMETRIC_COLLIDER_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct data_isometricCollider_s
{
   unsigned int bodyId;

   float restitution;
   // Collider's coefficient of friction
   float mu;
   unsigned int enabled;
} data_isometricCollider_t;

void initialize_isometricCollider(data_isometricCollider_t * data);

int isometricCollider_to_json(json_value_t * node, const data_isometricCollider_t * data);

int isometricCollider_from_json(const json_value_t * node, data_isometricCollider_t * data);

int anon_isometricCollider_to_json(json_value_t * node, const void * anon_data);

int anon_isometricCollider_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
