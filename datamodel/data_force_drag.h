#ifndef DATA_FORCE_DRAG_HEADER
#define DATA_FORCE_DRAG_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct data_forceDrag_s
{
   int childId;
   float linearDragCoeff;
   float quadraticDragCoeff;
} data_forceDrag_t;

void initialize_forceDrag(data_forceDrag_t * data);

int forceDrag_to_json(json_value_t * node, const data_forceDrag_t * data);

int forceDrag_from_json(const json_value_t * node, data_forceDrag_t * data);

int anon_forceDrag_to_json(json_value_t * node, const void * anon_data);

int anon_forceDrag_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
