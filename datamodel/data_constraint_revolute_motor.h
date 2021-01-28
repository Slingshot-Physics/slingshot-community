#ifndef DATA_CONSTRAINT_REVOLUTE_MOTOR_HEADER
#define DATA_CONSTRAINT_REVOLUTE_MOTOR_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct constraintRevoluteMotor
{
   int parentId;
   int childId;
   data_vector3_t parentAxis;
   data_vector3_t childAxis;
   float angularSpeed;
   float maxTorque;
} data_constraintRevoluteMotor_t;

void initialize_constraintRevoluteMotor(data_constraintRevoluteMotor_t * data);

int constraintRevoluteMotor_to_json(json_value_t * node, const data_constraintRevoluteMotor_t * data);

int constraintRevoluteMotor_from_json(const json_value_t * node, data_constraintRevoluteMotor_t * data);

int anon_constraintRevoluteMotor_to_json(json_value_t * node, const void * anon_data);

int anon_constraintRevoluteMotor_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
