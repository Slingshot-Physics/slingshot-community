#ifndef DATA_CONSTRAINT_GEAR_HEADER
#define DATA_CONSTRAINT_GEAR_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct constraintGear
{
   int parentId;
   int childId;
   float parentGearRadius;
   float childGearRadius;
   data_vector3_t parentAxis;
   data_vector3_t childAxis;
   unsigned int rotateParallel;
} data_constraintGear_t;

void initialize_constraintGear(data_constraintGear_t * data);

int constraintGear_to_json(json_value_t * node, const data_constraintGear_t * data);

int constraintGear_from_json(const json_value_t * node, data_constraintGear_t * data);

int anon_constraintGear_to_json(json_value_t * node, const void * anon_data);

int anon_constraintGear_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
