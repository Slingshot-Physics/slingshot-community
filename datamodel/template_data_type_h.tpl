#ifndef THIS_HEADER
#define THIS_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct data_<type>_s
{

} data_<type>_t;

void initialize_<type>(data_<type>_t * data);

int <type>_to_json(json_value_t * node, const data_<type>_t * data);

int <type>_from_json(const json_value_t * node, data_<type>_t * data);

int anon_<type>_to_json(json_value_t * node, const void * anon_data);

int anon_<type>_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
