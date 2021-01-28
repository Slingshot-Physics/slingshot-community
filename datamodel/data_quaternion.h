#ifndef DATA_QUATERNION_HEADER
#define DATA_QUATERNION_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct quaternion
{
   float q[4];
} data_quaternion_t;

void initialize_quaternion(data_quaternion_t * quat);

int quaternion_to_json(json_value_t * node, const data_quaternion_t * data);

int quaternion_from_json(const json_value_t * node, data_quaternion_t * data);

int anon_quaternion_to_json(json_value_t * node, const void * anon_data);

int anon_quaternion_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
