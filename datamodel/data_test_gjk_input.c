#include "data_test_gjk_input.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_testGjkInput(data_testGjkInput_t * data)
{
   memset(data, 0, sizeof(data_testGjkInput_t));
}

int testGjkInput_to_json(json_value_t * node, const data_testGjkInput_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_testGjkInput_t")) return 0;

   if (!add_object_field(node, "transformA", anon_transform_to_json, &(data->transformA))) return 0;
   if (!add_object_field(node, "shapeA", anon_shape_to_json, &(data->shapeA))) return 0;
   if (!add_object_field(node, "transformB", anon_transform_to_json, &(data->transformB))) return 0;
   if (!add_object_field(node, "shapeB", anon_shape_to_json, &(data->shapeB))) return 0;
   return 1;
}

int testGjkInput_from_json(const json_value_t * node, data_testGjkInput_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_testGjkInput_t")) return 0;

   if (!copy_object_field(node, "transformA", anon_transform_from_json, &(data->transformA))) return 0;
   if (!copy_object_field(node, "shapeA", anon_shape_from_json, &(data->shapeA))) return 0;
   if (!copy_object_field(node, "transformB", anon_transform_from_json, &(data->transformB))) return 0;
   if (!copy_object_field(node, "shapeB", anon_shape_from_json, &(data->shapeB))) return 0;
   return 1;
}

int anon_testGjkInput_to_json(json_value_t * node, const void * anon_data)
{
   const data_testGjkInput_t * data = (const data_testGjkInput_t *)anon_data;
   return testGjkInput_to_json(node, data);
}

int anon_testGjkInput_from_json(const json_value_t * node, void * anon_data)
{
   data_testGjkInput_t * data = (data_testGjkInput_t *)anon_data;
   return testGjkInput_from_json(node, data);
}
