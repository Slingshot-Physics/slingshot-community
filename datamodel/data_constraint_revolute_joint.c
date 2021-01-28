#include "data_constraint_revolute_joint.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_constraintRevoluteJoint(data_constraintRevoluteJoint_t * hinge)
{
   memset(hinge, 0, sizeof(data_constraintRevoluteJoint_t));
}

int constraintRevoluteJoint_to_json(json_value_t * node, const data_constraintRevoluteJoint_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_constraintRevoluteJoint_t")) return 0;
   if (!add_int_field(node, "parentId", data->parentId)) return 0;
   if (!add_int_field(node, "childId", data->childId)) return 0;
   
   if (!add_dynamic_object_array_field(node, "parentLinkPoints", 2, 2)) return 0;

   json_value_t * temp_lpa_field = get_field_by_name(node, "parentLinkPoints");
   if (temp_lpa_field->value_type != JSON_ARRAY) return 0;
   for (unsigned int i = 0; i < 2; ++i)
   {
      anon_vector3_to_json(&(temp_lpa_field->array.vals[i]), &(data->parentLinkPoints[i]));
   }

   if (!add_dynamic_object_array_field(node, "childLinkPoints", 2, 2)) return 0;

   json_value_t * temp_lpb_field = get_field_by_name(node, "childLinkPoints");
   if (temp_lpb_field->value_type != JSON_ARRAY) return 0;
   for (unsigned int i = 0; i < 2; ++i)
   {
      anon_vector3_to_json(&(temp_lpb_field->array.vals[i]), &(data->childLinkPoints[i]));
   }

   return 1;
}

int constraintRevoluteJoint_from_json(const json_value_t * node, data_constraintRevoluteJoint_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_constraintRevoluteJoint_t")) return 0;
   if (!copy_int_field(node, "parentId", &(data->parentId))) return 0;
   if (!copy_int_field(node, "childId", &(data->childId))) return 0;

   const json_value_t * temp_lpa_field = get_const_field_by_name(node, "parentLinkPoints");
   if (temp_lpa_field->value_type != JSON_ARRAY) return 0;
   for (unsigned int i = 0; i < 2; ++i)
   {
      anon_vector3_from_json(&(temp_lpa_field->array.vals[i]), &(data->parentLinkPoints[i]));
   }

   const json_value_t * temp_lpb_field = get_const_field_by_name(node, "childLinkPoints");
   if (temp_lpb_field->value_type != JSON_ARRAY) return 0;
   for (unsigned int i = 0; i < 2; ++i)
   {
      anon_vector3_from_json(&(temp_lpb_field->array.vals[i]), &(data->childLinkPoints[i]));
   }

   return 1;
}

int anon_constraintRevoluteJoint_to_json(json_value_t * node, const void * anon_data)
{
   const data_constraintRevoluteJoint_t * data = (const data_constraintRevoluteJoint_t *)anon_data;
   return constraintRevoluteJoint_to_json(node, data);
}

int anon_constraintRevoluteJoint_from_json(const json_value_t * node, void * anon_data)
{
   data_constraintRevoluteJoint_t * data = (data_constraintRevoluteJoint_t *)anon_data;
   return constraintRevoluteJoint_from_json(node, data);
}
