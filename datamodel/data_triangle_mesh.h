#ifndef DATA_TRIANGLE_MESH_HEADER
#define DATA_TRIANGLE_MESH_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_constants.h"
#include "data_mesh_triangle.h"
#include "data_vector3.h"

// Vertices, triangles, and normals forming a mesh.
typedef struct data_triangleMesh_s
{
   unsigned int numTriangles;
   data_meshTriangle_t triangles[DATA_MAX_TRIANGLES];
   unsigned int numVerts;
   data_vector3_t verts[DATA_MAX_VERTICES];
} data_triangleMesh_t;

void initialize_triangleMesh(data_triangleMesh_t * data);

int triangleMesh_to_json(json_value_t * node, const data_triangleMesh_t * data);

int triangleMesh_from_json(const json_value_t * node, data_triangleMesh_t * data);

int anon_triangleMesh_to_json(json_value_t * node, const void * anon_data);

int anon_triangleMesh_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
