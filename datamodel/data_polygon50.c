#include "data_polygon50.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_polygon50(data_polygon50_t * data)
{
   memset(data, 0, sizeof(data_polygon50_t));
}

int polygon50_to_json(json_value_t * node, const data_polygon50_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_polygon50_t")) return 0;
   if (!add_uint_field(node, "numVerts", data->numVerts)) return 0;
   if (!add_dynamic_object_array_field(node, "verts", data->numVerts, 50)) return 0;

   json_value_t * temp_vert_arr = get_field_by_name(node, "verts");
   if (temp_vert_arr == NULL) return 0;
   for (unsigned int i = 0; i < data->numVerts; ++i)
   {
      if (
         !anon_vector3_to_json(&(temp_vert_arr->array.vals[i]), &(data->verts[i]))
      ) return 0;
   }

   return 1;
}

int polygon50_from_json(const json_value_t * node, data_polygon50_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_polygon50_t")) return 0;
   if (!copy_uint_field(node, "numVerts", &(data->numVerts))) return 0;

   const json_value_t * temp_vert_arr = get_const_field_by_name(node, "verts");
   if (temp_vert_arr == NULL) return 0;
   for (unsigned int i = 0; i < data->numVerts; ++i)
   {
      if (
         !anon_vector3_from_json(&(temp_vert_arr->array.vals[i]), &(data->verts[i]))
      ) return 0;
   }
   return 1;
}

int anon_polygon50_to_json(json_value_t * node, const void * anon_data)
{
   const data_polygon50_t * data = (const data_polygon50_t *)anon_data;
   return polygon50_to_json(node, data);
}

int anon_polygon50_from_json(const json_value_t * node, void * anon_data)
{
   data_polygon50_t * data = (data_polygon50_t *)anon_data;
   return polygon50_from_json(node, data);
}
