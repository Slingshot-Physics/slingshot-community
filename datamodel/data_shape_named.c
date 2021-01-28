#include "data_shape_named.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_shapeNamed(data_shapeNamed_t * data)
{
   memset(data, 0, sizeof(data_shapeNamed_t));
}

int shapeNamed_to_json(json_value_t * node, const data_shapeNamed_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_shapeNamed_t")) return 0;
   if (!add_uint_field(node, "bodyId", data->bodyId)) return 0;
   if (!add_int_field(node, "shapeName", (int )data->shapeName)) return 0;

   return 1;
}

int shapeNamed_from_json(const json_value_t * node, data_shapeNamed_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_shapeNamed_t")) return 0;
   if (!copy_uint_field(node, "bodyId", &(data->bodyId))) return 0;
   int temp_int;
   if (!copy_int_field(node, "shapeName", &temp_int)) return 0;
   data->shapeName = (data_shapeType_t )temp_int;

   return 1;
}

int anon_shapeNamed_to_json(json_value_t * node, const void * anon_data)
{
   const data_shapeNamed_t * data = (const data_shapeNamed_t *)anon_data;
   return shapeNamed_to_json(node, data);
}

int anon_shapeNamed_from_json(const json_value_t * node, void * anon_data)
{
   data_shapeNamed_t * data = (data_shapeNamed_t *)anon_data;
   return shapeNamed_from_json(node, data);
}
