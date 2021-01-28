#include "data_shape_cylinder.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_shapeCylinder(data_shapeCylinder_t * data)
{
   memset(data, 0, sizeof(data_shapeCylinder_t));
}

int shapeCylinder_to_json(json_value_t * node, const data_shapeCylinder_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_shapeCylinder_t")) return 0;
   if (!add_float_field(node, "radius", data->radius)) return 0;
   if (!add_float_field(node, "height", data->height)) return 0;

   return 1;
}

int shapeCylinder_from_json(const json_value_t * node, data_shapeCylinder_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_shapeCylinder_t")) return 0;
   if (!copy_float_field(node, "radius", &(data->radius))) return 0;
   if (!copy_float_field(node, "height", &(data->height))) return 0;

   return 1;
}

int anon_shapeCylinder_to_json(json_value_t * node, const void * anon_data)
{
   const data_shapeCylinder_t * data = (const data_shapeCylinder_t *)anon_data;
   return shapeCylinder_to_json(node, data);
}

int anon_shapeCylinder_from_json(const json_value_t * node, void * anon_data)
{
   data_shapeCylinder_t * data = (data_shapeCylinder_t *)anon_data;
   return shapeCylinder_from_json(node, data);
}
