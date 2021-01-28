#ifndef DATA_RIGIDBODY_HEADER
#define DATA_RIGIDBODY_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_matrix33.h"
#include "data_quaternion.h"
#include "data_vector3.h"

#include <inttypes.h>

typedef struct rigidbody
{
   unsigned int id;
   data_matrix33_t J;
   data_vector3_t linpos;
   data_vector3_t linvel;
   data_quaternion_t orientation;
   data_vector3_t rotvel;
   unsigned int stationary;
   float mass;
} data_rigidbody_t;

void initialize_rigidbody(data_rigidbody_t * rb);

int rigidbody_to_json(json_value_t * node, const data_rigidbody_t * data);

int rigidbody_from_json(const json_value_t * node, data_rigidbody_t * data);

int anon_rigidbody_to_json(json_value_t * node, const void * anon_data);

int anon_rigidbody_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
