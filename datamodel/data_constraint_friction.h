#ifndef DATA_CONSTRAINT_FRICTION_HEADER
#define DATA_CONSTRAINT_FRICTION_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct constraintFriction
{
   // IDs of the bodies affected by the friction constraint
   int bodyIdA;
   int bodyIdB;

   // Contact points on the bodies in global coordinates relative to the
   // bodies' centers of mass in global coordinates.
   data_vector3_t bodyAContact;
   data_vector3_t bodyBContact;

   // Coefficient of friction
   float muTotal;

   // Contact normal
   data_vector3_t unitNormal;
} data_constraintFriction_t;

void initialize_constraintFriction(data_constraintFriction_t * data);

int constraintFriction_to_json(json_value_t * node, const data_constraintFriction_t * data);

int constraintFriction_from_json(const json_value_t * node, data_constraintFriction_t * data);

int anon_constraintFriction_to_json(json_value_t * node, const void * anon_data);

int anon_constraintFriction_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
