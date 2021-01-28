#ifndef SHAPE_NAMED_HEADER
#define SHAPE_NAMED_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_enums.h"

typedef struct shapeNamed
{
   unsigned int bodyId;
   data_shapeType_t shapeName;
} data_shapeNamed_t;

void initialize_shapeNamed(data_shapeNamed_t * data);

int shapeNamed_to_json(json_value_t * node, const data_shapeNamed_t * data);

int shapeNamed_from_json(const json_value_t * node, data_shapeNamed_t * data);

int anon_shapeNamed_to_json(json_value_t * node, const void * anon_data);

int anon_shapeNamed_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
