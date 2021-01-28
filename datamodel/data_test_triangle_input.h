#ifndef DATA_TEST_TRIANGLE_INPUT_HEADER
#define DATA_TEST_TRIANGLE_INPUT_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_triangle.h"
#include "data_vector3.h"

typedef struct data_testTriangleInput_s
{
   data_triangle_t triangle;
   data_vector3_t queryPointBary;
   data_vector3_t queryPoint;
} data_testTriangleInput_t;

void initialize_testTriangleInput(data_testTriangleInput_t * data);

int testTriangleInput_to_json(json_value_t * node, const data_testTriangleInput_t * data);

int testTriangleInput_from_json(const json_value_t * node, data_testTriangleInput_t * data);

int anon_testTriangleInput_to_json(json_value_t * node, const void * anon_data);

int anon_testTriangleInput_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
