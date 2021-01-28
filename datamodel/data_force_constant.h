#ifndef DATA_FORCE_CONSTANT_HEADER
#define DATA_FORCE_CONSTANT_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_enums.h"
#include "data_vector3.h"

typedef struct data_forceConstant_s
{
   int childId;
   data_vector3_t childLinkPoint;
   data_vector3_t acceleration;
   data_frameType_t frame;
} data_forceConstant_t;

void initialize_forceConstant(data_forceConstant_t * data);

int forceConstant_to_json(json_value_t * node, const data_forceConstant_t * data);

int forceConstant_from_json(const json_value_t * node, data_forceConstant_t * data);

int anon_forceConstant_to_json(json_value_t * node, const void * anon_data);

int anon_forceConstant_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
