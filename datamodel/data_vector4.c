#include "data_vector4.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_vector4(data_vector4_t * vec)
{
   memset(vec, 0, sizeof(data_vector4_t));
}

int vector4_to_json(json_value_t * node, const data_vector4_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_vector4_t")) return 0;
   if (!add_fixed_float_array_field(node, "v", 4, data->v)) return 0;
   return 1;
}

int vector4_from_json(const json_value_t * node, data_vector4_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_vector4_t")) return 0;
   if (!copy_fixed_float_array_field(node, "v", 4, data->v)) return 0;
   return 1;
}

int anon_vector4_to_json(json_value_t * node, const void * anon_data)
{
   const data_vector4_t * data = (const data_vector4_t *)anon_data;
   return vector4_to_json(node, data);
}

int anon_vector4_from_json(const json_value_t * node, void * anon_data)
{
   data_vector4_t * data = (data_vector4_t *)anon_data;
   return vector4_from_json(node, data);
}
