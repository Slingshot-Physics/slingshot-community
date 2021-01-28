#include "data_minkowski_diff_simplex.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_minkowskiDiffSimplex(data_minkowskiDiffSimplex_t * data)
{
   memset(data, 0, sizeof(data_minkowskiDiffSimplex_t));
}

int minkowskiDiffSimplex_to_json(json_value_t * node, const data_minkowskiDiffSimplex_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_minkowskiDiffSimplex_t")) return 0;

   if (!add_fixed_int_array_field(node, "bodyAVertIds", 4, &(data->bodyAVertIds[0]))) return 0;
   if (!add_fixed_int_array_field(node, "bodyBVertIds", 4, &(data->bodyBVertIds[0]))) return 0;

   if (!add_object_field(node, "minNormBary", anon_vector4_to_json, &(data->minNormBary))) return 0;
   if (!add_uint_field(node, "numVerts", data->numVerts)) return 0;
   if (!add_filled_dynamic_object_array_field(node, "verts", data->numVerts, 4, anon_vector3_to_json, &(data->verts[0]), sizeof(data->verts[0]))) return 0;
   if (!add_filled_dynamic_object_array_field(node, "bodyAVerts", data->numVerts, 4, anon_vector3_to_json, &(data->bodyAVerts[0]), sizeof(data->bodyAVerts[0]))) return 0;
   if (!add_filled_dynamic_object_array_field(node, "bodyBVerts", data->numVerts, 4, anon_vector3_to_json, &(data->bodyBVerts[0]), sizeof(data->bodyBVerts[0]))) return 0;
   return 1;
}

int minkowskiDiffSimplex_from_json(const json_value_t * node, data_minkowskiDiffSimplex_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_minkowskiDiffSimplex_t")) return 0;

   if (!copy_fixed_int_array_field(node, "bodyAVertIds", 4, &(data->bodyAVertIds[0]))) return 0;
   if (!copy_fixed_int_array_field(node, "bodyBVertIds", 4, &(data->bodyBVertIds[0]))) return 0;
   if (!copy_object_field(node, "minNormBary", anon_vector4_from_json, &(data->minNormBary))) return 0;
   if (!copy_uint_field(node, "numVerts", &(data->numVerts))) return 0;
   if (!copy_filled_dynamic_object_array_field(node, "verts", data->numVerts, 4, anon_vector3_from_json, &(data->verts[0]), sizeof(data->verts[0]))) return 0;
   if (!copy_filled_dynamic_object_array_field(node, "bodyAVerts", data->numVerts, 4, anon_vector3_from_json, &(data->bodyAVerts[0]), sizeof(data->bodyAVerts[0]))) return 0;
   if (!copy_filled_dynamic_object_array_field(node, "bodyBVerts", data->numVerts, 4, anon_vector3_from_json, &(data->bodyBVerts[0]), sizeof(data->bodyBVerts[0]))) return 0;
   return 1;
}

int anon_minkowskiDiffSimplex_to_json(json_value_t * node, const void * anon_data)
{
   const data_minkowskiDiffSimplex_t * data = (const data_minkowskiDiffSimplex_t *)anon_data;
   return minkowskiDiffSimplex_to_json(node, data);
}

int anon_minkowskiDiffSimplex_from_json(const json_value_t * node, void * anon_data)
{
   data_minkowskiDiffSimplex_t * data = (data_minkowskiDiffSimplex_t *)anon_data;
   return minkowskiDiffSimplex_from_json(node, data);
}
