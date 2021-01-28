#include "data_vector3.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_vector3(data_vector3_t * vec)
{
   memset(vec, 0, sizeof(data_vector3_t));
}

int vector3_to_json(json_value_t * node, const data_vector3_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_vector3_t")) return 0;
   if (!add_fixed_float_array_field(node, "v", 3, data->v)) return 0;
   return 1;
}

int vector3_from_json(const json_value_t * node, data_vector3_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_vector3_t")) return 0;
   if (!copy_fixed_float_array_field(node, "v", 3, &(data->v[0]))) return 0;
   return 1;
}

int anon_vector3_to_json(json_value_t * node, const void * anon_data)
{
   const data_vector3_t * data = (const data_vector3_t *)anon_data;
   return vector3_to_json(node, data);
}

int anon_vector3_from_json(const json_value_t * node, void * anon_data)
{
   data_vector3_t * data = (data_vector3_t *)anon_data;
   return vector3_from_json(node, data);
}
