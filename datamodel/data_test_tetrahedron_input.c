#include "data_test_tetrahedron_input.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_testTetrahedronInput(data_testTetrahedronInput_t * data)
{
   memset(data, 0, sizeof(data_testTetrahedronInput_t));
}

int testTetrahedronInput_to_json(json_value_t * node, const data_testTetrahedronInput_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_testTetrahedronInput_t")) return 0;
   if (!add_object_field(node, "tetrahedron", anon_tetrahedron_to_json, &(data->tetrahedron))) return 0;
   if (!add_object_field(node, "queryPointBary", anon_vector4_to_json, &(data->queryPointBary))) return 0;
   if (!add_object_field(node, "queryPoint", anon_vector3_to_json, &(data->queryPoint))) return 0;
   return 1;
}

int testTetrahedronInput_from_json(const json_value_t * node, data_testTetrahedronInput_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_testTetrahedronInput_t")) return 0;
   if (!copy_object_field(node, "tetrahedron", anon_tetrahedron_from_json, &(data->tetrahedron))) return 0;
   if (!copy_object_field(node, "queryPointBary", anon_vector4_from_json, &(data->queryPointBary))) return 0;
   if (!copy_object_field(node, "queryPoint", anon_vector3_from_json, &(data->queryPoint))) return 0;
   return 1;
}

int anon_testTetrahedronInput_to_json(json_value_t * node, const void * anon_data)
{
   const data_testTetrahedronInput_t * data = (const data_testTetrahedronInput_t *)anon_data;
   return testTetrahedronInput_to_json(node, data);
}

int anon_testTetrahedronInput_from_json(const json_value_t * node, void * anon_data)
{
   data_testTetrahedronInput_t * data = (data_testTetrahedronInput_t *)anon_data;
   return testTetrahedronInput_from_json(node, data);
}
