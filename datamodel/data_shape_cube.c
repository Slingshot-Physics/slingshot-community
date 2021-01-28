#include "data_shape_cube.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_shapeCube(data_shapeCube_t * data)
{
   memset(data, 0, sizeof(data_shapeCube_t));
}

int shapeCube_to_json(json_value_t * node, const data_shapeCube_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_shapeCube_t")) return 0;
   if (!add_float_field(node, "length", data->length)) return 0;
   if (!add_float_field(node, "width", data->width)) return 0;
   if (!add_float_field(node, "height", data->height)) return 0;
   return 1;
}

int shapeCube_from_json(const json_value_t * node, data_shapeCube_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_shapeCube_t")) return 0;
   if (!copy_float_field(node, "length", &(data->length))) return 0;
   if (!copy_float_field(node, "width", &(data->width))) return 0;
   if (!copy_float_field(node, "height", &(data->height))) return 0;
   return 1;
}

int anon_shapeCube_to_json(json_value_t * node, const void * anon_data)
{
   const data_shapeCube_t * data = (const data_shapeCube_t *)anon_data;
   return shapeCube_to_json(node, data);
}

int anon_shapeCube_from_json(const json_value_t * node, void * anon_data)
{
   data_shapeCube_t * data = (data_shapeCube_t *)anon_data;
   return shapeCube_from_json(node, data);
}
