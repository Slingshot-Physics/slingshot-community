#ifndef DATA_CONSTRAINT_TRANSLATION_1D_HEADER
#define DATA_CONSTRAINT_TRANSLATION_1D_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct data_constraintTranslation1d_s
{
   int parentId;
   int childId;

   // A vector in the parent's body frame defining the normal of the
   // constraint plane in the parent's body frame.
   data_vector3_t parentAxis;

   // A point in the parent's body frame that is on the constraint plane in
   // the parent's body frame.
   data_vector3_t parentLinkPoint;

   // A point in the child's body frame that is constrained to live on the
   // plane defined in the parent's body frame.
   data_vector3_t childLinkPoint;
} data_constraintTranslation1d_t;

void initialize_constraintTranslation1d(data_constraintTranslation1d_t * data);

int constraintTranslation1d_to_json(json_value_t * node, const data_constraintTranslation1d_t * data);

int constraintTranslation1d_from_json(const json_value_t * node, data_constraintTranslation1d_t * data);

int anon_constraintTranslation1d_to_json(json_value_t * node, const void * anon_data);

int anon_constraintTranslation1d_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
