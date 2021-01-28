#include "data_lean_neighbor_triangle.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_leanNeighborTriangle(data_leanNeighborTriangle_t * data)
{
   memset(data, 0, sizeof(data_leanNeighborTriangle_t));
}

int leanNeighborTriangle_to_json(json_value_t * node, const data_leanNeighborTriangle_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_leanNeighborTriangle_t")) return 0;
   if (!add_fixed_int_array_field(node, "vertIds", 3, &(data->vertIds[0]))) return 0;
   if (!add_fixed_int_array_field(node, "neighborIds", 3, &(data->neighborIds[0]))) return 0;
   return 1;
}

int leanNeighborTriangle_from_json(const json_value_t * node, data_leanNeighborTriangle_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_leanNeighborTriangle_t")) return 0;
   if (!copy_fixed_int_array_field(node, "vertIds", 3, &(data->vertIds[0]))) return 0;
   if (!copy_fixed_int_array_field(node, "neighborIds", 3, &(data->neighborIds[0]))) return 0;
   return 1;
}

int anon_leanNeighborTriangle_to_json(json_value_t * node, const void * anon_data)
{
   const data_leanNeighborTriangle_t * data = (const data_leanNeighborTriangle_t *)anon_data;
   return leanNeighborTriangle_to_json(node, data);
}

int anon_leanNeighborTriangle_from_json(const json_value_t * node, void * anon_data)
{
   data_leanNeighborTriangle_t * data = (data_leanNeighborTriangle_t *)anon_data;
   return leanNeighborTriangle_from_json(node, data);
}
