#include "data_isometric_transform.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_isometricTransform(data_isometricTransform_t * data)
{
   memset(data, 0, sizeof(data_isometricTransform_t));
}

int isometricTransform_to_json(json_value_t * node, const data_isometricTransform_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_isometricTransform_t")) return 0;
   if (!add_object_field(node, "rotate", anon_matrix33_to_json, &(data->rotate))) return 0;
   if (!add_object_field(node, "translate", anon_vector3_to_json, &(data->translate))) return 0;
   return 1;
}

int isometricTransform_from_json(const json_value_t * node, data_isometricTransform_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_isometricTransform_t")) return 0;
   if (!copy_object_field(node, "rotate", anon_matrix33_from_json, &(data->rotate))) return 0;
   if (!copy_object_field(node, "translate", anon_vector3_from_json, &(data->translate))) return 0;
   return 1;
}

int anon_isometricTransform_to_json(json_value_t * node, const void * anon_data)
{
   const data_isometricTransform_t * data = (const data_isometricTransform_t *)anon_data;
   return isometricTransform_to_json(node, data);
}

int anon_isometricTransform_from_json(const json_value_t * node, void * anon_data)
{
   data_isometricTransform_t * data = (data_isometricTransform_t *)anon_data;
   return isometricTransform_from_json(node, data);
}
