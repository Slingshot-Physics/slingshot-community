#ifndef DATA_CONSTRAINTBALLJOINT_HEADER
#define DATA_CONSTRAINTBALLJOINT_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"

typedef struct constraintBalljoint
{
   int parentId;
   int childId;
   // Positions relative to CM in body frame that are linked.
   data_vector3_t parentLinkPoint;
   data_vector3_t childLinkPoint;
} data_constraintBalljoint_t;

void initialize_constraintBalljoint(data_constraintBalljoint_t * joint);

int constraintBalljoint_to_json(
   json_value_t * node, const data_constraintBalljoint_t * data
);

int constraintBalljoint_from_json(
   const json_value_t * node, data_constraintBalljoint_t * data
);

int anon_constraintBalljoint_to_json(
   json_value_t * node, const void * anon_data
);

int anon_constraintBalljoint_from_json(
   const json_value_t * node, void * anon_data
);

#ifdef __cplusplus
}
#endif

#endif
