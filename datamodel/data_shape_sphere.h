#ifndef DATA_SHAPESPHERE_HEADER
#define DATA_SHAPESPHERE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct shapeSphere
{
   float radius;
} data_shapeSphere_t;

void initialize_shapeSphere(data_shapeSphere_t * data);

int shapeSphere_to_json(json_value_t * node, const data_shapeSphere_t * data);

int shapeSphere_from_json(const json_value_t * node, data_shapeSphere_t * data);

int anon_shapeSphere_to_json(json_value_t * node, const void * anon_data);

int anon_shapeSphere_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
