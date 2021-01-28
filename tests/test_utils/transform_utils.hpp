#ifndef TRANSFORM_UTILS_HEADER
#define TRANSFORM_UTILS_HEADER

#include "geometry_types.hpp"

namespace test_utils
{

   geometry::types::transform_t generate_random_transform(void);

   geometry::types::transform_t generate_random_scale_transform(void);
   
   // Vertex indices from global mesh and convex polyhedron are lined up.
   void convert_triangleMesh_to_convexPolyhedron(
      const geometry::types::triangleMesh_t & data_in,
      geometry::types::convexPolyhedron_t & data_out
   );

}

#endif
