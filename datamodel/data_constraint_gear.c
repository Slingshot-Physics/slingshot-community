#include "data_constraint_gear.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

#include <stdbool.h>

void initialize_constraintGear(data_constraintGear_t * data)
{
   memset(data, 0, sizeof(data_constraintGear_t));
}

int constraintGear_to_json(json_value_t * node, const data_constraintGear_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_constraintGear_t")) return 0;
   if (!add_int_field(node, "parentId", data->parentId)) return 0;
   if (!add_int_field(node, "childId", data->childId)) return 0;
   if (!add_float_field(node, "parentGearRadius", data->parentGearRadius)) return 0;
   if (!add_float_field(node, "childGearRadius", data->childGearRadius)) return 0;
   if (!add_object_field(node, "parentAxis", anon_vector3_to_json, &(data->parentAxis))) return 0;
   if (!add_object_field(node, "childAxis", anon_vector3_to_json, &(data->childAxis))) return 0;
   if (!add_uint_field(node, "rotateParallel", data->rotateParallel)) return 0;

   return 1;
}

int constraintGear_from_json(const json_value_t * node, data_constraintGear_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_constraintGear_t")) return 0;
   if (!copy_int_field(node, "parentId", &(data->parentId))) return 0;
   if (!copy_int_field(node, "childId", &(data->childId))) return 0;
   if (!copy_float_field(node, "parentGearRadius", &(data->parentGearRadius))) return 0;
   if (!copy_float_field(node, "childGearRadius", &(data->childGearRadius))) return 0;
   if (!copy_object_field(node, "parentAxis", anon_vector3_from_json, &(data->parentAxis))) return 0;
   if (!copy_object_field(node, "childAxis", anon_vector3_from_json, &(data->childAxis))) return 0;
   if (!copy_uint_field(node, "rotateParallel", &(data->rotateParallel))) return 0;

   return 1;
}

int anon_constraintGear_to_json(json_value_t * node, const void * anon_data)
{
   const data_constraintGear_t * data = (const data_constraintGear_t *)anon_data;
   return constraintGear_to_json(node, data);
}

int anon_constraintGear_from_json(const json_value_t * node, void * anon_data)
{
   data_constraintGear_t * data = (data_constraintGear_t *)anon_data;
   return constraintGear_from_json(node, data);
}
