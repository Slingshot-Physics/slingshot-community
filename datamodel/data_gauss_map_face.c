#include "data_gauss_map_face.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_gaussMapFace(data_gaussMapFace_t * data)
{
   memset(data, 0, sizeof(data_gaussMapFace_t));
}

int gaussMapFace_to_json(json_value_t * node, const data_gaussMapFace_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_gaussMapFace_t")) return 0;
   if (!add_object_field(node, "normal", anon_vector3_to_json, &(data->normal))) return 0;
   if (!add_uint_field(node, "triangleStartId", data->triangleStartId)) return 0;
   if (!add_uint_field(node, "numTriangles", data->numTriangles)) return 0;
   return 1;
}

int gaussMapFace_from_json(const json_value_t * node, data_gaussMapFace_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_gaussMapFace_t")) return 0;
   if (!copy_object_field(node, "normal", anon_vector3_from_json, &(data->normal))) return 0;
   if (!copy_uint_field(node, "triangleStartId", &(data->triangleStartId))) return 0;
   if (!copy_uint_field(node, "numTriangles", &(data->numTriangles))) return 0;
   return 1;
}

int anon_gaussMapFace_to_json(json_value_t * node, const void * anon_data)
{
   const data_gaussMapFace_t * data = (const data_gaussMapFace_t *)anon_data;
   return gaussMapFace_to_json(node, data);
}

int anon_gaussMapFace_from_json(const json_value_t * node, void * anon_data)
{
   data_gaussMapFace_t * data = (data_gaussMapFace_t *)anon_data;
   return gaussMapFace_from_json(node, data);
}
