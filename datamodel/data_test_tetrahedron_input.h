#ifndef DATA_TEST_TETRAHEDRON_INPUT_HEADER
#define DATA_TEST_TETRAHEDRON_INPUT_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_tetrahedron.h"
#include "data_vector3.h"
#include "data_vector4.h"

typedef struct data_testTetrahedronInput_s
{
   data_tetrahedron_t tetrahedron;
   data_vector4_t queryPointBary;
   data_vector3_t queryPoint;
} data_testTetrahedronInput_t;

void initialize_testTetrahedronInput(data_testTetrahedronInput_t * data);

int testTetrahedronInput_to_json(json_value_t * node, const data_testTetrahedronInput_t * data);

int testTetrahedronInput_from_json(const json_value_t * node, data_testTetrahedronInput_t * data);

int anon_testTetrahedronInput_to_json(json_value_t * node, const void * anon_data);

int anon_testTetrahedronInput_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
