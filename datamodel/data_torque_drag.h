#ifndef DATA_TORQUE_DRAG_HEADER
#define DATA_TORQUE_DRAG_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct data_torqueDrag_s
{
   int childId;

   // Drag coefficient for angular velocity, should be less than or equal to
   // zero.
   float linearDragCoeff;

   // Drag coefficient for quadratic angular velocity, should be less than
   // or equal to zero.
   float quadraticDragCoeff;
} data_torqueDrag_t;

void initialize_torqueDrag(data_torqueDrag_t * data);

int torqueDrag_to_json(json_value_t * node, const data_torqueDrag_t * data);

int torqueDrag_from_json(const json_value_t * node, data_torqueDrag_t * data);

int anon_torqueDrag_to_json(json_value_t * node, const void * anon_data);

int anon_torqueDrag_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
