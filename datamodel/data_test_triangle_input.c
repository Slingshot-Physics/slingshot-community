#include "data_test_triangle_input.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_testTriangleInput(data_testTriangleInput_t * data)
{
   memset(data, 0, sizeof(data_testTriangleInput_t));
}

int testTriangleInput_to_json(json_value_t * node, const data_testTriangleInput_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_testTriangleInput_t")) return 0;
   if (!add_object_field(node, "triangle", anon_triangle_to_json, &(data->triangle))) return 0;
   if (!add_object_field(node, "queryPointBary", anon_vector3_to_json, &(data->queryPointBary))) return 0;
   if (!add_object_field(node, "queryPoint", anon_vector3_to_json, &(data->queryPoint))) return 0;
   return 1;
}

int testTriangleInput_from_json(const json_value_t * node, data_testTriangleInput_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_testTriangleInput_t")) return 0;
   if (!copy_object_field(node, "triangle", anon_triangle_from_json, &(data->triangle))) return 0;
   if (!copy_object_field(node, "queryPointBary", anon_vector3_from_json, &(data->queryPointBary))) return 0;
   if (!copy_object_field(node, "queryPoint", anon_vector3_from_json, &(data->queryPoint))) return 0;
   return 1;
}

int anon_testTriangleInput_to_json(json_value_t * node, const void * anon_data)
{
   const data_testTriangleInput_t * data = (const data_testTriangleInput_t *)anon_data;
   return testTriangleInput_to_json(node, data);
}

int anon_testTriangleInput_from_json(const json_value_t * node, void * anon_data)
{
   data_testTriangleInput_t * data = (data_testTriangleInput_t *)anon_data;
   return testTriangleInput_from_json(node, data);
}
