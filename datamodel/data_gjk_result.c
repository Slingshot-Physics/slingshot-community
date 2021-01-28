#include "data_gjk_result.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_gjkResult(data_gjkResult_t * data)
{
   memset(data, 0, sizeof(data_gjkResult_t));
}

int gjkResult_to_json(json_value_t * node, const data_gjkResult_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_gjkResult_t")) return 0;

   if (!add_object_field(node, "minSimplex", anon_minkowskiDiffSimplex_to_json, &(data->minSimplex))) return 0;
   if (!add_uint_field(node, "intersection", data->intersection)) return 0;
   return 1;
}

int gjkResult_from_json(const json_value_t * node, data_gjkResult_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_gjkResult_t")) return 0;

   if (!copy_object_field(node, "minSimplex", anon_minkowskiDiffSimplex_from_json, &(data->minSimplex))) return 0;
   if (!copy_uint_field(node, "intersection", &(data->intersection))) return 0;
   return 1;
}

int anon_gjkResult_to_json(json_value_t * node, const void * anon_data)
{
   const data_gjkResult_t * data = (const data_gjkResult_t *)anon_data;
   return gjkResult_to_json(node, data);
}

int anon_gjkResult_from_json(const json_value_t * node, void * anon_data)
{
   data_gjkResult_t * data = (data_gjkResult_t *)anon_data;
   return gjkResult_from_json(node, data);
}
