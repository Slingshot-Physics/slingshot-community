#ifndef DATA_TEST_GJK_INPUT_HEADER
#define DATA_TEST_GJK_INPUT_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_shape.h"
#include "data_transform.h"

typedef struct data_testGjkInput_s
{
   data_transform_t transformA;
   data_shape_t shapeA;
   data_transform_t transformB;
   data_shape_t shapeB;
} data_testGjkInput_t;

void initialize_testGjkInput(data_testGjkInput_t * data);

int testGjkInput_to_json(json_value_t * node, const data_testGjkInput_t * data);

int testGjkInput_from_json(const json_value_t * node, data_testGjkInput_t * data);

int anon_testGjkInput_to_json(json_value_t * node, const void * anon_data);

int anon_testGjkInput_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
