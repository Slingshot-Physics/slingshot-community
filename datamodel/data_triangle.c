#include "data_triangle.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_triangle(data_triangle_t * data)
{
   memset(data, 0, sizeof(data_triangle_t));
}

int triangle_to_json(json_value_t * node, const data_triangle_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_triangle_t")) return 0;
   if (!add_filled_dynamic_object_array_field(node, "verts", 3, 3, anon_vector3_to_json, &(data->verts[0]), sizeof(data->verts[0]))) return 0;
   return 1;
}

int triangle_from_json(const json_value_t * node, data_triangle_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_triangle_t")) return 0;
   if (!copy_filled_dynamic_object_array_field(node, "verts", 3, 3, anon_vector3_from_json, &(data->verts[0]), sizeof(data->verts[0]))) return 0;
   return 1;
}

int anon_triangle_to_json(json_value_t * node, const void * anon_data)
{
   const data_triangle_t * data = (const data_triangle_t *)anon_data;
   return triangle_to_json(node, data);
}

int anon_triangle_from_json(const json_value_t * node, void * anon_data)
{
   data_triangle_t * data = (data_triangle_t *)anon_data;
   return triangle_from_json(node, data);
}
