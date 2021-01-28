#include "data_multi_collider.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_multiCollider(data_multiCollider_t * data)
{
   memset(data, 0, sizeof(data_multiCollider_t));
}

int multiCollider_to_json(json_value_t * node, const data_multiCollider_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_multiCollider_t")) return 0;

   if (!add_uint_field(node, "numColliders", data->numColliders)) return 0;

   if (
      !add_filled_dynamic_object_array_field(
         node,
         "colliders",
         data->numColliders,
         64,
         anon_isometricCollider_to_json,
         &(data->colliders[0]),
         sizeof(data->colliders[0])
      )
   ) return 0;

   return 1;
}

int multiCollider_from_json(const json_value_t * node, data_multiCollider_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_multiCollider_t")) return 0;

   if (!copy_uint_field(node, "numColliders", &(data->numColliders))) return 0;

   if (
      !copy_filled_dynamic_object_array_field(
         node,
         "colliders",
         data->numColliders,
         64,
         anon_isometricCollider_from_json,
         &(data->colliders[0]),
         sizeof(data->colliders[0])
      )
   ) return 0;

   return 1;
}

int anon_multiCollider_to_json(json_value_t * node, const void * anon_data)
{
   const data_multiCollider_t * data = (const data_multiCollider_t *)anon_data;
   return multiCollider_to_json(node, data);
}

int anon_multiCollider_from_json(const json_value_t * node, void * anon_data)
{
   data_multiCollider_t * data = (data_multiCollider_t *)anon_data;
   return multiCollider_from_json(node, data);
}
