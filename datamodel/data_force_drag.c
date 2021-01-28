#include "data_force_drag.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_forceDrag(data_forceDrag_t * data)
{
   memset(data, 0, sizeof(data_forceDrag_t));
}

int forceDrag_to_json(json_value_t * node, const data_forceDrag_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_forceDrag_t")) return 0;
   if (!add_int_field(node, "childId", data->childId)) return 0;
   if (!add_float_field(node, "linearDragCoeff", data->linearDragCoeff)) return 0;
   if (!add_float_field(node, "quadraticDragCoeff", data->quadraticDragCoeff)) return 0;
   return 1;
}

int forceDrag_from_json(const json_value_t * node, data_forceDrag_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_forceDrag_t")) return 0;
   if (!copy_int_field(node, "childId", &(data->childId))) return 0;
   if (!copy_float_field(node, "linearDragCoeff", &(data->linearDragCoeff))) return 0;
   if (!copy_float_field(node, "quadraticDragCoeff", &(data->quadraticDragCoeff))) return 0;
   return 1;
}

int anon_forceDrag_to_json(json_value_t * node, const void * anon_data)
{
   const data_forceDrag_t * data = (const data_forceDrag_t *)anon_data;
   return forceDrag_to_json(node, data);
}

int anon_forceDrag_from_json(const json_value_t * node, void * anon_data)
{
   data_forceDrag_t * data = (data_forceDrag_t *)anon_data;
   return forceDrag_from_json(node, data);
}
