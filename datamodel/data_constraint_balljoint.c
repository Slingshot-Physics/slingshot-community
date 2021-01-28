#include "data_constraint_balljoint.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_constraintBalljoint(data_constraintBalljoint_t * joint)
{
   memset(joint, 0, sizeof(data_constraintBalljoint_t));
}

int constraintBalljoint_to_json(
   json_value_t * node, const data_constraintBalljoint_t * data
)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_constraintBalljoint_t")) return 0;
   if (!add_int_field(node, "parentId", data->parentId)) return 0;
   if (!add_int_field(node, "childId", data->childId)) return 0;
   if (!add_object_field(node, "parentLinkPoint", anon_vector3_to_json, &(data->parentLinkPoint))) return 0;
   if (!add_object_field(node, "childLinkPoint", anon_vector3_to_json, &(data->childLinkPoint))) return 0;
   return 1;
}

int constraintBalljoint_from_json(
   const json_value_t * node, data_constraintBalljoint_t * data
)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_constraintBalljoint_t")) return 0;
   if (!copy_int_field(node, "parentId", &(data->parentId))) return 0;
   if (!copy_int_field(node, "childId", &(data->childId))) return 0;
   if (!copy_object_field(node, "parentLinkPoint", anon_vector3_from_json, &(data->parentLinkPoint))) return 0;
   if (!copy_object_field(node, "childLinkPoint", anon_vector3_from_json, &(data->childLinkPoint))) return 0;
   return 1;
}

int anon_constraintBalljoint_to_json(
   json_value_t * node, const void * anon_data
)
{
   const data_constraintBalljoint_t * data = (const data_constraintBalljoint_t *)anon_data;
   return constraintBalljoint_to_json(node, data);
}

int anon_constraintBalljoint_from_json(
   const json_value_t * node, void * anon_data
)
{
   data_constraintBalljoint_t * data = (data_constraintBalljoint_t *)anon_data;
   return constraintBalljoint_from_json(node, data);
}
