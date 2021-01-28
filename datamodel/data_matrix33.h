#ifndef DATA_MATRIX33_HEADER
#define DATA_MATRIX33_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct matrix33
{
   float m[3][3];
} data_matrix33_t;

void initialize_matrix33(data_matrix33_t * mat);

int matrix33_to_json(json_value_t * node, const data_matrix33_t * data);

int matrix33_from_json(const json_value_t * node, data_matrix33_t * data);

int anon_matrix33_to_json(json_value_t * node, const void * anon_data);

int anon_matrix33_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
