#include "data_rigidbody.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_rigidbody(data_rigidbody_t * rb)
{
   memset(rb, 0, sizeof(data_rigidbody_t));
}

int rigidbody_to_json(json_value_t * node, const data_rigidbody_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_rigidbody_t")) return 0;
   if (!add_uint_field(node, "id", data->id)) return 0;
   if (!add_object_field(node, "J", anon_matrix33_to_json, &(data->J))) return 0;
   if (!add_object_field(node, "linpos", anon_vector3_to_json, &(data->linpos))) return 0;
   if (!add_object_field(node, "linvel", anon_vector3_to_json, &(data->linvel))) return 0;
   if (!add_object_field(node, "orientation", anon_quaternion_to_json, &(data->orientation))) return 0;
   if (!add_object_field(node, "rotvel", anon_vector3_to_json, &(data->rotvel))) return 0;
   if (!add_uint_field(node, "stationary", data->stationary)) return 0;
   if (!add_float_field(node, "mass", data->mass)) return 0;

   return 1;
}

int rigidbody_from_json(const json_value_t * node, data_rigidbody_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_rigidbody_t")) return 0;
   if (!copy_uint_field(node, "id", &(data->id))) return 0;
   if (!copy_object_field(node, "J", anon_matrix33_from_json, &(data->J))) return 0;
   if (!copy_object_field(node, "linpos", anon_vector3_from_json, &(data->linpos))) return 0;
   if (!copy_object_field(node, "linvel", anon_vector3_from_json, &(data->linvel))) return 0;
   if (!copy_object_field(node, "orientation", anon_quaternion_from_json, &(data->orientation))) return 0;
   if (!copy_object_field(node, "rotvel", anon_vector3_from_json, &(data->rotvel))) return 0;
   if (!copy_uint_field(node, "stationary", &(data->stationary))) return 0;
   if (!copy_float_field(node, "mass", &(data->mass))) return 0;

   return 1;
}

int anon_rigidbody_to_json(json_value_t * node, const void * anon_data)
{
   const data_rigidbody_t * data = (const data_rigidbody_t *)anon_data;
   return rigidbody_to_json(node, data);
}

int anon_rigidbody_from_json(const json_value_t * node, void * anon_data)
{
   data_rigidbody_t * data = (data_rigidbody_t *)anon_data;
   return rigidbody_from_json(node, data);
}
