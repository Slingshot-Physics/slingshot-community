#include "data_quaternion.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_quaternion(data_quaternion_t * quat)
{
   memset(quat, 0, sizeof(data_quaternion_t));
}

int quaternion_to_json(json_value_t * node, const data_quaternion_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_quaternion_t")) return 0;
   if (!add_fixed_float_array_field(node, "q", 4, data->q)) return 0;
   return 1;
}

int quaternion_from_json(const json_value_t * node, data_quaternion_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_quaternion_t")) return 0;
   if (!copy_fixed_float_array_field(node, "q", 4, data->q)) return 0;
   return 1;
}

int anon_quaternion_to_json(json_value_t * node, const void * anon_data)
{
   const data_quaternion_t * data = (const data_quaternion_t *)anon_data;
   return quaternion_to_json(node, data);
}

int anon_quaternion_from_json(const json_value_t * node, void * anon_data)
{
   data_quaternion_t * data = (data_quaternion_t *)anon_data;
   return quaternion_from_json(node, data);
}
