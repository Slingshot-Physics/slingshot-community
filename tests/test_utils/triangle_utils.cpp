#include "triangle_utils.hpp"

#include "random_utils.hpp"

#include <iostream>

namespace test_utils
{
   geometry::types::triangle_t generate_random_xy_triangle(void)
   {
      geometry::types::triangle_t output;

      for (int i = 0; i < 2; ++i)
      {
         float theta = edbdmath::random_float(-1.f * M_PI, M_PI);
         float radius = edbdmath::random_float(0.1f, 32.f);
         output.verts[i].Initialize(radius * cosf(theta), radius * sinf(theta), 0.f);
      }

      Vector3 points_unit = (output.verts[0] - output.verts[1]).unitVector();

      float theta = edbdmath::random_float(0.001f * M_PI, M_PI);
      float sign = edbdmath::random_float() > 0.5f ? 1.f : -1.f;
      float radius = edbdmath::random_float(0.1f, 32.f);

      Matrix33 rot(
         cosf(theta), sinf(theta), 0.f, -sinf(theta), cosf(theta), 0.f, 0.f, 0.f, 1.f
      );

      output.verts[2] = rot * points_unit * sign * radius;

      return output;
   }

   geometry::types::triangle_t generate_random_xy_triangle(
      float min_radius, float max_radius
   )
   {
      geometry::types::triangle_t output;

      if (min_radius >= max_radius)
      {
         std::cout << "wtf your random triangle min radius is larger than the max radius\n";
         return output;
      }

      for (int i = 0; i < 2; ++i)
      {
         float theta = edbdmath::random_float(-1.f * M_PI, M_PI);
         float radius = edbdmath::random_float(min_radius, max_radius);
         output.verts[i].Initialize(radius * cosf(theta), radius * sinf(theta), 0.f);
      }

      Vector3 points_unit = (output.verts[0] - output.verts[1]).unitVector();

      float theta = edbdmath::random_float(0.001f * M_PI, M_PI);
      float sign = edbdmath::random_float() > 0.5f ? 1.f : -1.f;
      float radius = edbdmath::random_float(min_radius, max_radius);

      Matrix33 rot(
         cosf(theta), sinf(theta), 0.f, -sinf(theta), cosf(theta), 0.f, 0.f, 0.f, 1.f
      );

      output.verts[2] = rot * points_unit * sign * radius;

      return output;
   }
}
