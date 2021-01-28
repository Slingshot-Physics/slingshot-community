#include "data_shape_sphere.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_shapeSphere(data_shapeSphere_t * data)
{
   memset(data, 0, sizeof(data_shapeSphere_t));
}

int shapeSphere_to_json(json_value_t * node, const data_shapeSphere_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_shapeSphere_t")) return 0;
   if (!add_float_field(node, "radius", data->radius)) return 0;

   return 1;
}

int shapeSphere_from_json(const json_value_t * node, data_shapeSphere_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_shapeSphere_t")) return 0;
   if (!copy_float_field(node, "radius", &(data->radius))) return 0;

   return 1;
}

int anon_shapeSphere_to_json(json_value_t * node, const void * anon_data)
{
   const data_shapeSphere_t * data = (const data_shapeSphere_t *)anon_data;
   return shapeSphere_to_json(node, data);
}

int anon_shapeSphere_from_json(const json_value_t * node, void * anon_data)
{
   data_shapeSphere_t * data = (data_shapeSphere_t *)anon_data;
   return shapeSphere_from_json(node, data);
}
