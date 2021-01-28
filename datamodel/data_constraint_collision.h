#ifndef DATA_CONSTRAINTCOLLISION_HEADER
#define DATA_CONSTRAINTCOLLISION_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct constraintCollision
{
   // Collision normal
   data_vector3_t n_hat;

   int bodyIdA;
   data_vector3_t bodyAContact;

   int bodyIdB;
   data_vector3_t bodyBContact;

   float restitution;

} data_constraintCollision_t;

void initialize_constraintCollision(data_constraintCollision_t * collision);

int constraintCollision_to_json(json_value_t * node, const data_constraintCollision_t * data);

int constraintCollision_from_json(const json_value_t * node, data_constraintCollision_t * data);

int anon_constraintCollision_to_json(json_value_t * node, const void * anon_data);

int anon_constraintCollision_from_json(const json_value_t * node, void * anon_data);
#ifdef __cplusplus
}
#endif

#endif
