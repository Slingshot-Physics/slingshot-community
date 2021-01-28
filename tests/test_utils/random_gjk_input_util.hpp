#ifndef RANDOM_GJK_INPUT_UTIL_HEADER
#define RANDOM_GJK_INPUT_UTIL_HEADER

#include "geometry_types.hpp"

namespace test_utils
{
   // Takes a given GJK input file and loads the respective polyhedra and
   // meshes. Polyhedra are scaled according to the transformation in 'data'.
   // Meshes have the full transforms from data applied to them.
   void load_meshes_and_polyhedra(
      const geometry::types::testGjkInput_t & data,
      geometry::types::convexPolyhedron_t & polyhedronA,
      geometry::types::convexPolyhedron_t & polyhedronB,
      geometry::types::triangleMesh_t & meshA,
      geometry::types::triangleMesh_t & meshB
   );

   // Generates random transforms for two cube shape types. Loads the cube
   // meshes and polyhedra. The polyhedra are scaled according to the randomly
   // generated transform. The meshes have the complete transforms applied to
   // them.
   void generate_random_gjk_input(
      const geometry::types::enumShape_t shapeAType,
      const geometry::types::enumShape_t shapeBType,
      geometry::types::testGjkInput_t & data,
      geometry::types::convexPolyhedron_t & polyhedronA,
      geometry::types::convexPolyhedron_t & polyhedronB,
      geometry::types::triangleMesh_t & meshA,
      geometry::types::triangleMesh_t & meshB
   );

}

#endif
