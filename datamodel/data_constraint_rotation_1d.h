#ifndef DATA_CONSTRAINT_ROTATION_1D_HEADER
#define DATA_CONSTRAINT_ROTATION_1D_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct data_constraintRotation1d_s
{
   int parentId;
   int childId;

   // A unit vector in the parent's body frame.
   data_vector3_t parentAxis;

   // A unit vector in the child's body frame.
   data_vector3_t childAxis;
} data_constraintRotation1d_t;

void initialize_constraintRotation1d(data_constraintRotation1d_t * data);

int constraintRotation1d_to_json(json_value_t * node, const data_constraintRotation1d_t * data);

int constraintRotation1d_from_json(const json_value_t * node, data_constraintRotation1d_t * data);

int anon_constraintRotation1d_to_json(json_value_t * node, const void * anon_data);

int anon_constraintRotation1d_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
