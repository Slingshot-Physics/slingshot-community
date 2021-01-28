#ifndef EPA_TYPES_HEADER
#define EPA_TYPES_HEADER

#include "geometry_types.hpp"
#include "triangle_graph_mesh.hpp"
#include "vector3.hpp"
#include "vector4.hpp"

#include "data_triangle_mesh.h"

namespace geometry
{

namespace types
{

namespace epa
{
   #define MAX_SIZE 128

   typedef geometry::mesh::TriangleGraphMesh<MAX_SIZE> EpaTriangles;

   typedef geometry::types::labeledVertex_t epa_labeledVertex_t;

   typedef geometry::types::minkowskiDiffVertex_t epa_mdVert_t;

   struct epaMesh_t
   {
      EpaTriangles triangles;

      geometry::types::minkowskiDiffVertex_t mdVerts[3 * MAX_SIZE];

      unsigned int numMdVerts;

      epaMesh_t(void);

      epaMesh_t(const geometry::types::minkowskiDiffSimplex_t & tetra_simplex);

      void initialize(const geometry::types::minkowskiDiffSimplex_t & tetra_simplex);

      void print(void) const;

      void print(unsigned int id) const;

      bool vertExists(const geometry::types::minkowskiDiffVertex_t & md_vert);

      // Returns -2 if the vertex cannot be added due to size restrictions.
      // Returns the new number of vertices in the mesh if the vertex is added.
      int addVert(const geometry::types::minkowskiDiffVertex_t & md_vert);

      void to_triangle_mesh(data_triangleMesh_t * data_out);

   };

} // epa

} // types

} // geometry

#endif
