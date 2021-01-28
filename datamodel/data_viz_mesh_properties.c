#include "data_viz_mesh_properties.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_vizMeshProperties(data_vizMeshProperties_t * data)
{
   memset(data, 0, sizeof(data_vizMeshProperties_t));
}

int vizMeshProperties_to_json(
   json_value_t * node, const data_vizMeshProperties_t * data
)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_vizMeshProperties_t")) return 0;
   if (!add_uint_field(node, "bodyId", data->bodyId)) return 0;
   if (!add_object_field(node, "color", anon_vector4_to_json, &(data->color))) return 0;

   return 1;
}

int vizMeshProperties_from_json(
   const json_value_t * node, data_vizMeshProperties_t * data
)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_vizMeshProperties_t")) return 0;
   if (!copy_uint_field(node, "bodyId", &(data->bodyId))) return 0;
   if (!copy_object_field(node, "color", anon_vector4_from_json, &(data->color))) return 0;

   return 1;
}

int anon_vizMeshProperties_to_json(json_value_t * node, const void * anon_data)
{
   const data_vizMeshProperties_t * data = (const data_vizMeshProperties_t *)anon_data;
   return vizMeshProperties_to_json(node, data);
}

int anon_vizMeshProperties_from_json(const json_value_t * node, void * anon_data)
{
   data_vizMeshProperties_t * data = (data_vizMeshProperties_t *)anon_data;
   return vizMeshProperties_from_json(node, data);
}
