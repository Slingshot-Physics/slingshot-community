#ifndef DATA_LEAN_NEIGHBOR_TRIANGLE_HEADER
#define DATA_LEAN_NEIGHBOR_TRIANGLE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

typedef struct data_leanNeighborTriangle_s
{
   // Indices of the vertices in the mesh contributing to this triangle.
   int vertIds[3];
   // Indices/IDs of the three triangles sharing sides with this triangle.
   // Refers to triangles in the MeshTriangleGraph class.
   int neighborIds[3];
} data_leanNeighborTriangle_t;

void initialize_leanNeighborTriangle(data_leanNeighborTriangle_t * data);

int leanNeighborTriangle_to_json(json_value_t * node, const data_leanNeighborTriangle_t * data);

int leanNeighborTriangle_from_json(const json_value_t * node, data_leanNeighborTriangle_t * data);

int anon_leanNeighborTriangle_to_json(json_value_t * node, const void * anon_data);

int anon_leanNeighborTriangle_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
