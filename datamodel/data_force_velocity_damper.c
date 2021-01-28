#include "data_force_velocity_damper.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_forceVelocityDamper(data_forceVelocityDamper_t * data)
{
   memset(data, 0, sizeof(data_forceVelocityDamper_t));
}

int forceVelocityDamper_to_json(json_value_t * node, const data_forceVelocityDamper_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_forceVelocityDamper_t")) return 0;
   if (!add_int_field(node, "parentId", data->parentId)) return 0;
   if (!add_int_field(node, "childId", data->childId)) return 0;
   if (!add_float_field(node, "damperCoeff", data->damperCoeff)) return 0;
   if (!add_object_field(node, "parentLinkPoint", anon_vector3_to_json, &(data->parentLinkPoint))) return 0;
   if (!add_object_field(node, "childLinkPoint", anon_vector3_to_json, &(data->childLinkPoint))) return 0;

   return 1;
}

int forceVelocityDamper_from_json(const json_value_t * node, data_forceVelocityDamper_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_forceVelocityDamper_t")) return 0;
   if (!copy_int_field(node, "parentId", &(data->parentId))) return 0;
   if (!copy_int_field(node, "childId", &(data->childId))) return 0;
   if (!copy_float_field(node, "damperCoeff", &(data->damperCoeff))) return 0;
   if (!copy_object_field(node, "parentLinkPoint", anon_vector3_from_json, &(data->parentLinkPoint))) return 0;
   if (!copy_object_field(node, "childLinkPoint", anon_vector3_from_json, &(data->childLinkPoint))) return 0;

   return 1;
}

int anon_forceVelocityDamper_to_json(json_value_t * node, const void * anon_data)
{
   const data_forceVelocityDamper_t * data = (const data_forceVelocityDamper_t *)anon_data;
   return forceVelocityDamper_to_json(node, data);
}

int anon_forceVelocityDamper_from_json(const json_value_t * node, void * anon_data)
{
   data_forceVelocityDamper_t * data = (data_forceVelocityDamper_t *)anon_data;
   return forceVelocityDamper_from_json(node, data);
}
