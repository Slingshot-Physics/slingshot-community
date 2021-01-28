#include "data_matrix33.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

int matrix33_to_json(json_value_t * node, const data_matrix33_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_matrix33_t")) return 0;
   float m[9] = {
      data->m[0][0], data->m[0][1], data->m[0][2],
      data->m[1][0], data->m[1][1], data->m[1][2],
      data->m[2][0], data->m[2][1], data->m[2][2]
   };
   if (!add_fixed_float_array_field(node, "m", 9, m)) return 0;
   return 1;
}

int matrix33_from_json(const json_value_t * node, data_matrix33_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_matrix33_t")) return 0;
   float m[9];
   if (!copy_fixed_float_array_field(node, "m", 9, m)) return 0;
   data->m[0][0] = m[0];
   data->m[0][1] = m[1];
   data->m[0][2] = m[2];
   data->m[1][0] = m[3];
   data->m[1][1] = m[4];
   data->m[1][2] = m[5];
   data->m[2][0] = m[6];
   data->m[2][1] = m[7];
   data->m[2][2] = m[8];
   return 1;
}

int anon_matrix33_to_json(json_value_t * node, const void * anon_data)
{
   const data_matrix33_t * data = (const data_matrix33_t *)anon_data;
   return matrix33_to_json(node, data);
}

int anon_matrix33_from_json(const json_value_t * node, void * anon_data)
{
   data_matrix33_t * data = (data_matrix33_t *)anon_data;
   return matrix33_from_json(node, data);
}
