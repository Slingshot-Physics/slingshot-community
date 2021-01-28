#ifndef TRIANGLE_UTILS_HEADER
#define TRIANGLE_UTILS_HEADER

#include "geometry_types.hpp"

namespace test_utils
{
   // Creates a non-degenerate triangle in the xy plane.
   geometry::types::triangle_t generate_random_xy_triangle(void);

   geometry::types::triangle_t generate_random_xy_triangle(
      float min_radius, float max_radius
   );
}

#endif
