#include "data_constraint_friction.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_constraintFriction(data_constraintFriction_t * data)
{
   memset(data, 0, sizeof(data_constraintFriction_t));
}

int constraintFriction_to_json(json_value_t * node, const data_constraintFriction_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_constraintFriction_t")) return 0;
   if (!add_int_field(node, "bodyIdA", data->bodyIdA)) return 0;
   if (!add_int_field(node, "bodyIdB", data->bodyIdB)) return 0;
   if (!add_object_field(node, "bodyAContact", anon_vector3_to_json, &(data->bodyAContact))) return 0;
   if (!add_object_field(node, "bodyBContact", anon_vector3_to_json, &(data->bodyBContact))) return 0;
   if (!add_float_field(node, "muTotal", data->muTotal)) return 0;
   if (!add_object_field(node, "unitNormal", anon_vector3_to_json, &(data->unitNormal))) return 0;

   return 1;
}

int constraintFriction_from_json(const json_value_t * node, data_constraintFriction_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_constraintFriction_t")) return 0;
   if (!copy_int_field(node, "bodyIdA", &(data->bodyIdA))) return 0;
   if (!copy_int_field(node, "bodyIdB", &(data->bodyIdB))) return 0;
   if (!copy_object_field(node, "bodyAContact", anon_vector3_from_json, &(data->bodyAContact))) return 0;
   if (!copy_object_field(node, "bodyBContact", anon_vector3_from_json, &(data->bodyBContact))) return 0;
   if (!copy_float_field(node, "muTotal", &(data->muTotal))) return 0;
   if (!copy_object_field(node, "unitNormal", anon_vector3_from_json, &(data->unitNormal))) return 0;

   return 1;
}

int anon_constraintFriction_to_json(json_value_t * node, const void * anon_data)
{
   const data_constraintFriction_t * data = (const data_constraintFriction_t *)anon_data;
   return constraintFriction_to_json(node, data);
}

int anon_constraintFriction_from_json(const json_value_t * node, void * anon_data)
{
   data_constraintFriction_t * data = (data_constraintFriction_t *)anon_data;
   return constraintFriction_from_json(node, data);
}
