#ifndef DATA_MULTI_COLLIDER_HEADER
#define DATA_MULTI_COLLIDER_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_isometric_collider.h"

typedef struct data_multiCollider_s
{
   unsigned int numColliders;
   data_isometricCollider_t colliders[64];
} data_multiCollider_t;

void initialize_multiCollider(data_multiCollider_t * data);

int multiCollider_to_json(json_value_t * node, const data_multiCollider_t * data);

int multiCollider_from_json(const json_value_t * node, data_multiCollider_t * data);

int anon_multiCollider_to_json(json_value_t * node, const void * anon_data);

int anon_multiCollider_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
