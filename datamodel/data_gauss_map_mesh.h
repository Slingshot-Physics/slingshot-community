#ifndef DATA_GAUSS_MAP_MESH_HEADER
#define DATA_GAUSS_MAP_MESH_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_constants.h"
#include "data_gauss_map_face.h"
#include "data_lean_neighbor_triangle.h"
#include "data_vector3.h"

typedef struct data_gaussMapMesh_s
{
   unsigned int numTriangles;
   data_leanNeighborTriangle_t triangles[DATA_MAX_TRIANGLES];
   unsigned int numVerts;
   data_vector3_t verts[DATA_MAX_VERTICES];
   unsigned int numFaces;
   data_gaussMapFace_t faces[DATA_MAX_TRIANGLES];
} data_gaussMapMesh_t;

void initialize_gaussMapMesh(data_gaussMapMesh_t * data);

int gaussMapMesh_to_json(json_value_t * node, const data_gaussMapMesh_t * data);

int gaussMapMesh_from_json(const json_value_t * node, data_gaussMapMesh_t * data);

int anon_gaussMapMesh_to_json(json_value_t * node, const void * anon_data);

int anon_gaussMapMesh_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
