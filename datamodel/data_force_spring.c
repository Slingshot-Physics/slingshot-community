#include "data_force_spring.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_forceSpring(data_forceSpring_t * data)
{
   memset(data, 0, sizeof(data_forceSpring_t));
}

int forceSpring_to_json(json_value_t * node, const data_forceSpring_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_forceSpring_t")) return 0;
   if (!add_int_field(node, "parentId", data->parentId)) return 0;
   if (!add_int_field(node, "childId", data->childId)) return 0;
   if (!add_float_field(node, "restLength", data->restLength)) return 0;
   if (!add_float_field(node, "springCoeff", data->springCoeff)) return 0;
   if (!add_object_field(node, "parentLinkPoint", anon_vector3_to_json, &(data->parentLinkPoint))) return 0;
   if (!add_object_field(node, "childLinkPoint", anon_vector3_to_json, &(data->childLinkPoint))) return 0;

   return 1;
}

int forceSpring_from_json(const json_value_t * node, data_forceSpring_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_forceSpring_t")) return 0;
   if (!copy_int_field(node, "parentId", &(data->parentId))) return 0;
   if (!copy_int_field(node, "childId", &(data->childId))) return 0;
   if (!copy_float_field(node, "restLength", &(data->restLength))) return 0;
   if (!copy_float_field(node, "springCoeff", &(data->springCoeff))) return 0;
   if (!copy_object_field(node, "parentLinkPoint", anon_vector3_from_json, &(data->parentLinkPoint))) return 0;
   if (!copy_object_field(node, "childLinkPoint", anon_vector3_from_json, &(data->childLinkPoint))) return 0;

   return 1;
}

int anon_forceSpring_to_json(json_value_t * node, const void * anon_data)
{
   const data_forceSpring_t * data = (const data_forceSpring_t *)anon_data;
   return forceSpring_to_json(node, data);
}

int anon_forceSpring_from_json(const json_value_t * node, void * anon_data)
{
   data_forceSpring_t * data = (data_forceSpring_t *)anon_data;
   return forceSpring_from_json(node, data);
}
