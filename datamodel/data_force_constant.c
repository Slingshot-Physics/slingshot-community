#include "data_force_constant.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_forceConstant(data_forceConstant_t * data)
{
   memset(data, 0, sizeof(data_forceConstant_t));
}

int forceConstant_to_json(json_value_t * node, const data_forceConstant_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_forceConstant_t")) return 0;
   if (!add_int_field(node, "childId", data->childId)) return 0;
   if (!add_object_field(node, "childLinkPoint", anon_vector3_to_json, &(data->childLinkPoint))) return 0;
   if (!add_object_field(node, "acceleration", anon_vector3_to_json, &(data->acceleration))) return 0;
   int temp_frame = (int )data->frame;
   if (!add_int_field(node, "frame", temp_frame)) return 0;

   return 1;
}

int forceConstant_from_json(const json_value_t * node, data_forceConstant_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_forceConstant_t")) return 0;
   if (!copy_int_field(node, "childId", &(data->childId))) return 0;
   if (!copy_object_field(node, "childLinkPoint", anon_vector3_from_json, &(data->childLinkPoint))) return 0;
   if (!copy_object_field(node, "acceleration", anon_vector3_from_json, &(data->acceleration))) return 0;
   int temp_frame = 0;
   if (!copy_int_field(node, "frame", &temp_frame)) return 0;
   data->frame = (data_frameType_t )temp_frame;

   return 1;
}

int anon_forceConstant_to_json(json_value_t * node, const void * anon_data)
{
   const data_forceConstant_t * data = (const data_forceConstant_t *)anon_data;
   return forceConstant_to_json(node, data);
}

int anon_forceConstant_from_json(const json_value_t * node, void * anon_data)
{
   data_forceConstant_t * data = (data_forceConstant_t *)anon_data;
   return forceConstant_from_json(node, data);
}
