#include "data_gauss_map_mesh.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_gaussMapMesh(data_gaussMapMesh_t * data)
{
   memset(data, 0, sizeof(data_gaussMapMesh_t));
}

int gaussMapMesh_to_json(json_value_t * node, const data_gaussMapMesh_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_gaussMapMesh_t")) return 0;
   if (!add_uint_field(node, "numTriangles", data->numTriangles)) return 0;
   if (!add_uint_field(node, "numVerts", data->numVerts)) return 0;
   if (!add_uint_field(node, "numFaces", data->numFaces)) return 0;

   if (
      !add_filled_dynamic_object_array_field(
         node,
         "triangles",
         data->numTriangles,
         DATA_MAX_TRIANGLES,
         anon_leanNeighborTriangle_to_json,
         &(data->triangles[0]),
         sizeof(data->triangles[0])
      )
   ) return 0;

   if (
      !add_filled_dynamic_object_array_field(
         node,
         "verts",
         data->numVerts,
         DATA_MAX_VERTICES,
         anon_vector3_to_json,
         &(data->verts[0]),
         sizeof(data->verts[0])
      )
   ) return 0;

   if (
      !add_filled_dynamic_object_array_field(
         node,
         "faces",
         data->numFaces,
         DATA_MAX_TRIANGLES,
         anon_gaussMapFace_to_json,
         &(data->faces[0]),
         sizeof(data->faces[0])
      )
   ) return 0;

   return 1;
}

int gaussMapMesh_from_json(const json_value_t * node, data_gaussMapMesh_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_gaussMapMesh_t")) return 0;
   if (!copy_uint_field(node, "numTriangles", &(data->numTriangles))) return 0;
   if (!copy_uint_field(node, "numVerts", &(data->numVerts))) return 0;
   if (!copy_uint_field(node, "numFaces", &(data->numFaces))) return 0;

   if (
      !copy_filled_dynamic_object_array_field(
         node,
         "triangles",
         data->numTriangles,
         DATA_MAX_TRIANGLES,
         anon_leanNeighborTriangle_from_json,
         &(data->triangles[0]),
         sizeof(data->triangles[0])
      )
   ) return 0;

   if (
      !copy_filled_dynamic_object_array_field(
         node,
         "verts",
         data->numVerts,
         DATA_MAX_VERTICES,
         anon_vector3_from_json,
         &(data->verts[0]),
         sizeof(data->verts[0])
      )
   ) return 0;

   if (
      !copy_filled_dynamic_object_array_field(
         node,
         "faces",
         data->numFaces,
         DATA_MAX_TRIANGLES,
         anon_gaussMapFace_from_json,
         &(data->faces[0]),
         sizeof(data->faces[0])
      )
   ) return 0;

   return 1;
}

int anon_gaussMapMesh_to_json(json_value_t * node, const void * anon_data)
{
   const data_gaussMapMesh_t * data = (const data_gaussMapMesh_t *)anon_data;
   return gaussMapMesh_to_json(node, data);
}

int anon_gaussMapMesh_from_json(const json_value_t * node, void * anon_data)
{
   data_gaussMapMesh_t * data = (data_gaussMapMesh_t *)anon_data;
   return gaussMapMesh_from_json(node, data);
}
