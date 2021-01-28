#include "data_constraint_translation_1d.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_constraintTranslation1d(data_constraintTranslation1d_t * data)
{
   memset(data, 0, sizeof(data_constraintTranslation1d_t));
}

int constraintTranslation1d_to_json(json_value_t * node, const data_constraintTranslation1d_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_constraintTranslation1d_t")) return 0;
   if (!add_int_field(node, "parentId", data->parentId)) return 0;
   if (!add_int_field(node, "childId", data->childId)) return 0;
   if (!add_object_field(node, "parentAxis", anon_vector3_to_json, &(data->parentAxis))) return 0;
   if (!add_object_field(node, "parentLinkPoint", anon_vector3_to_json, &(data->parentLinkPoint))) return 0;
   if (!add_object_field(node, "childLinkPoint", anon_vector3_to_json, &(data->childLinkPoint))) return 0;
   return 1;
}

int constraintTranslation1d_from_json(const json_value_t * node, data_constraintTranslation1d_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_constraintTranslation1d_t")) return 0;
   if (!copy_int_field(node, "parentId", &(data->parentId))) return 0;
   if (!copy_int_field(node, "childId", &(data->childId))) return 0;
   if (!copy_object_field(node, "parentAxis", anon_vector3_from_json, &(data->parentAxis))) return 0;
   if (!copy_object_field(node, "parentLinkPoint", anon_vector3_from_json, &(data->parentLinkPoint))) return 0;
   if (!copy_object_field(node, "childLinkPoint", anon_vector3_from_json, &(data->childLinkPoint))) return 0;
   return 1;
}

int anon_constraintTranslation1d_to_json(json_value_t * node, const void * anon_data)
{
   const data_constraintTranslation1d_t * data = (const data_constraintTranslation1d_t *)anon_data;
   return constraintTranslation1d_to_json(node, data);
}

int anon_constraintTranslation1d_from_json(const json_value_t * node, void * anon_data)
{
   data_constraintTranslation1d_t * data = (data_constraintTranslation1d_t *)anon_data;
   return constraintTranslation1d_from_json(node, data);
}
