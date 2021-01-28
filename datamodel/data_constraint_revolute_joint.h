#ifndef DATA_CONSTRAINT_REVOLUTE_JOINT_HEADER
#define DATA_CONSTRAINT_REVOLUTE_JOINT_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct constraintRevoluteJoint
{
   int parentId;
   int childId;
   // Attachment points relative to each body's center of mass.
   data_vector3_t parentLinkPoints[2];
   data_vector3_t childLinkPoints[2];
} data_constraintRevoluteJoint_t;

void initialize_constraintRevoluteJoint(data_constraintRevoluteJoint_t * hinge);

int constraintRevoluteJoint_to_json(json_value_t * node, const data_constraintRevoluteJoint_t * data);

int constraintRevoluteJoint_from_json(const json_value_t * node, data_constraintRevoluteJoint_t * data);

int anon_constraintRevoluteJoint_to_json(json_value_t * node, const void * anon_data);

int anon_constraintRevoluteJoint_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
