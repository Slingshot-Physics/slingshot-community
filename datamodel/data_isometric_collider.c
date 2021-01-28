#include "data_isometric_collider.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_isometricCollider(data_isometricCollider_t * data)
{
   memset(data, 0, sizeof(data_isometricCollider_t));
}

int isometricCollider_to_json(json_value_t * node, const data_isometricCollider_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_isometricCollider_t")) return 0;
   if (!add_uint_field(node, "bodyId", data->bodyId)) return 0;
   if (!add_float_field(node, "restitution", data->restitution)) return 0;
   if (!add_float_field(node, "mu", data->mu)) return 0;
   if (!add_uint_field(node, "enabled", data->enabled)) return 0;
   return 1;
}

int isometricCollider_from_json(const json_value_t * node, data_isometricCollider_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_isometricCollider_t")) return 0;
   if (!copy_uint_field(node, "bodyId", &(data->bodyId))) return 0;
   if (!copy_float_field(node, "restitution", &(data->restitution))) return 0;
   if (!copy_float_field(node, "mu", &(data->mu))) return 0;
   if (!copy_uint_field(node, "enabled", &(data->enabled))) return 0;
   return 1;
}

int anon_isometricCollider_to_json(json_value_t * node, const void * anon_data)
{
   const data_isometricCollider_t * data = (const data_isometricCollider_t *)anon_data;
   return isometricCollider_to_json(node, data);
}

int anon_isometricCollider_from_json(const json_value_t * node, void * anon_data)
{
   data_isometricCollider_t * data = (data_isometricCollider_t *)anon_data;
   return isometricCollider_from_json(node, data);
}
