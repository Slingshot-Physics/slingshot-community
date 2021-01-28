#include "data_shape_capsule.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_shapeCapsule(data_shapeCapsule_t * data)
{
   memset(data, 0, sizeof(data_shapeCapsule_t));
}

int shapeCapsule_to_json(json_value_t * node, const data_shapeCapsule_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_shapeCapsule_t")) return 0;
   if (!add_float_field(node, "radius", data->radius)) return 0;
   if (!add_float_field(node, "height", data->height)) return 0;

   return 1;
}

int shapeCapsule_from_json(const json_value_t * node, data_shapeCapsule_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_shapeCapsule_t")) return 0;
   if (!copy_float_field(node, "radius", &(data->radius))) return 0;
   if (!copy_float_field(node, "height", &(data->height))) return 0;

   return 1;
}

int anon_shapeCapsule_to_json(json_value_t * node, const void * anon_data)
{
   const data_shapeCapsule_t * data = (const data_shapeCapsule_t *)anon_data;
   return shapeCapsule_to_json(node, data);
}

int anon_shapeCapsule_from_json(const json_value_t * node, void * anon_data)
{
   data_shapeCapsule_t * data = (data_shapeCapsule_t *)anon_data;
   return shapeCapsule_from_json(node, data);
}
