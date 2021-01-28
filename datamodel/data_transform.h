#ifndef DATA_TRANSFORM_HEADER
#define DATA_TRANSFORM_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_matrix33.h"
#include "data_vector3.h"

typedef struct data_transform_s
{
   data_matrix33_t scale;
   data_matrix33_t rotate;
   data_vector3_t translate;
} data_transform_t;

void initialize_transform(data_transform_t * data);

int transform_to_json(json_value_t * node, const data_transform_t * data);

int transform_from_json(const json_value_t * node, data_transform_t * data);

int anon_transform_to_json(json_value_t * node, const void * anon_data);

int anon_transform_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
