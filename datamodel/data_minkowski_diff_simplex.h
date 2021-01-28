#ifndef DATA_MINKOWSKI_DIFF_SIMPLEX_HEADER
#define DATA_MINKOWSKI_DIFF_SIMPLEX_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"
#include "data_vector4.h"

typedef struct data_minkowskiDiffSimplex_s
{
   int bodyAVertIds[4];

   int bodyBVertIds[4];

   data_vector4_t minNormBary;

   unsigned int numVerts;

   data_vector3_t verts[4];

   data_vector3_t bodyAVerts[4];

   data_vector3_t bodyBVerts[4];
} data_minkowskiDiffSimplex_t;

void initialize_minkowskiDiffSimplex(data_minkowskiDiffSimplex_t * data);

int minkowskiDiffSimplex_to_json(json_value_t * node, const data_minkowskiDiffSimplex_t * data);

int minkowskiDiffSimplex_from_json(const json_value_t * node, data_minkowskiDiffSimplex_t * data);

int anon_minkowskiDiffSimplex_to_json(json_value_t * node, const void * anon_data);

int anon_minkowskiDiffSimplex_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
