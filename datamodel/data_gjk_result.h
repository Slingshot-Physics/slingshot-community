#ifndef DATA_GJK_RESULT_HEADER
#define DATA_GJK_RESULT_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_minkowski_diff_simplex.h"

typedef struct data_gjkResult_s
{
   data_minkowskiDiffSimplex_t minSimplex;
   unsigned int intersection;
} data_gjkResult_t;

void initialize_gjkResult(data_gjkResult_t * data);

int gjkResult_to_json(json_value_t * node, const data_gjkResult_t * data);

int gjkResult_from_json(const json_value_t * node, data_gjkResult_t * data);

int anon_gjkResult_to_json(json_value_t * node, const void * anon_data);

int anon_gjkResult_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
