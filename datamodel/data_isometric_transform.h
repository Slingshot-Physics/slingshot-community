#ifndef DATA_ISOMETRIC_TRANSFORM_HEADER
#define DATA_ISOMETRIC_TRANSFORM_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_matrix33.h"
#include "data_vector3.h"

typedef struct data_isometricTransform_s
{
   data_matrix33_t rotate;
   data_vector3_t translate;
} data_isometricTransform_t;

void initialize_isometricTransform(data_isometricTransform_t * data);

int isometricTransform_to_json(json_value_t * node, const data_isometricTransform_t * data);

int isometricTransform_from_json(const json_value_t * node, data_isometricTransform_t * data);

int anon_isometricTransform_to_json(json_value_t * node, const void * anon_data);

int anon_isometricTransform_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
