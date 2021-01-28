#include "data_mesh_triangle.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_meshTriangle(data_meshTriangle_t * triangle)
{
   memset(triangle, 0, sizeof(data_meshTriangle_t));
}

int meshTriangle_to_json(json_value_t * node, const data_meshTriangle_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_meshTriangle_t")) return 0;
   if (!add_fixed_uint_array_field(node, "vertIds", 3, data->vertIds)) return 0;
   if (!add_object_field(node, "normal", anon_vector3_to_json, &(data->normal))) return 0;
   return 1;
}

int meshTriangle_from_json(const json_value_t * node, data_meshTriangle_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_meshTriangle_t")) return 0;
   if (!copy_fixed_uint_array_field(node, "vertIds", 3, &(data->vertIds[0]))) return 0;
   if (!copy_object_field(node, "normal", anon_vector3_from_json, &(data->normal))) return 0;
   return 1;
}

int anon_meshTriangle_to_json(json_value_t * node, const void * anon_data)
{
   const data_meshTriangle_t * data = (const data_meshTriangle_t *)anon_data;
   return meshTriangle_to_json(node, data);
}

int anon_meshTriangle_from_json(const json_value_t * node, void * anon_data)
{
   data_meshTriangle_t * data = (data_meshTriangle_t *)anon_data;
   return meshTriangle_from_json(node, data);
}
