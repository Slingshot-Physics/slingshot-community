// Calculations of intersection for quadratic shapes are made by writing out
// the equation of the quadratic surface in terms of collider coordinates and
// substituting the transformation from collider coordinates to world
// coordinates into the surface expression. E.g.:
//
//    f( p_C ) = p_C.dot( p_C ) - R ^ 2 = 0
//
//    p_W = R_{W/C} * S_{CW} * p_C + t_{W/C} = A * p_C + t_{W/C}
// 
//    p_C = inv( A ) * (p_W - t_{W/C})
//
//    f( p_C ) = f( inv(A) * ( p_W - t_{W/C} ) )
//
//    Where p_C is some point in collider space, and p_W is p_C transformed
//    into world coordinates.
//
// But we also know that p_W = a_W + b_W * u, where u is a scalar, a_W is the
// ray's start position, and b_W is the ray's slope.
//
// For the transformation equations, R_{W/C} is the rotation matrix that
// rotates from collider to world coordinates, S_{CW} is the scale matrix that
// operates on the original collider vertices, and t_{W/C} is the translation
// of the shape's center of geometry from collider to world coordinates.
// So substituting p_W(u) into the equation above gives you a quadratic
// equation in u, which can be solved analytically.
//
// Most of the complexity from this code arises from enumerating constraints
// on the acceptable values of 'u'. For a ray, u >= 0. Other shapes might have
// other requirements.

#include "raycast.hpp"

#include "transform.hpp"
#include "quadratic.hpp"

#include <algorithm>

namespace geometry
{
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeSphere_t & sphere_C
   )
   {
      const Vector3 ray_slope(ray_end - ray_start);
      const Matrix33 A_inv((~trans_C_to_W.scale) * trans_C_to_W.rotate.transpose());
      const Vector3 q_C(A_inv * (ray_start - trans_C_to_W.translate));
      const Vector3 y_C(A_inv * ray_slope);

      const float q_2 = q_C.magnitudeSquared();
      const float y_dot_q = y_C.dot(q_C);
      const float y_2 = y_C.magnitudeSquared();

      // a * u ^ 2 + b * u + c = 0
      edbdmath::quadraticResult_t quad_result = edbdmath::quadraticRoots(
         y_2, 2.f * y_dot_q, q_2 - sphere_C.radius * sphere_C.radius
      );

      geometry::types::raycastResult_t result;
      if (quad_result.numRealSolutions == 0)
      {
         result.numHits = 0;
         result.hit = false;
      }
      else if (quad_result.numRealSolutions == 1)
      {
         if (quad_result.x[0] >= 0.f)
         {
            result.hit = true;
            result.numHits = quad_result.numRealSolutions;
            result.hits[0] = quad_result.x[0] * ray_slope + ray_start;
            result.u[0] = quad_result.x[0];
         }
         else
         {
            result.numHits = 0;
            result.hit = false;
         }
      }
      else
      {
         unsigned int & num_hits = result.numHits;
         num_hits = 0;

         for (unsigned int i = 0; i < quad_result.numRealSolutions; ++i)
         {
            Vector3 intersection = quad_result.x[i] * ray_slope + ray_start;
            if (quad_result.x[i] >= 0.f)
            {
               result.hits[num_hits] = intersection;
               result.u[num_hits] = quad_result.x[i];
               ++num_hits;
            }
         }

         result.hit = (num_hits > 0);
      }

      return result;
   }

   // The equation for a cylinder is given as:
   //
   //    g(p_C) = (p_C ^ T) * H * p_C - (R ^ 2) = 0
   //
   //       abs(p_C[2] <= cylinder_C.height / 2
   //
   //       H = diag(1.f, 1.f, 0.f)
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCylinder_t & cylinder_C
   )
   {
      const Vector3 ray_slope(ray_end - ray_start);
      const Matrix33 A_inv((~trans_C_to_W.scale) * trans_C_to_W.rotate.transpose());
      const Vector3 q_C(A_inv * (ray_start - trans_C_to_W.translate));
      const Vector3 y_C(A_inv * ray_slope);
      const Matrix33 H(1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f);

      const Vector3 H_dot_y = H * y_C;
      const float a = y_C.dot(H_dot_y);
      const float b = 2.f * q_C.dot(H_dot_y);
      const float c = q_C.dot(H * q_C) - cylinder_C.radius * cylinder_C.radius;

      // a * u ^ 2 + b * u + c = 0
      // The solution is the intersection between the line through
      // [ray_start, ray_end] and the infinite cylinder.
      edbdmath::quadraticResult_t quad_result = edbdmath::quadraticRoots(a, b, c);

      // The maximum and minimum bounds on u based on the cylinder's height.
      const float u_bound1 = (-1.f * cylinder_C.height / 2 - q_C[2]) / (y_C[2]);
      const float u_bound2 = (cylinder_C.height / 2 - q_C[2]) / (y_C[2]);

      // Min and max values of u based on the bounds from the cap.
      const float u_min = std::min(u_bound1, u_bound2);
      const float u_max = std::max(u_bound1, u_bound2);

      const bool ray_inside_top_cap = \
         (edbdmath::quadraticEval(a, b, c, u_max) <= 0.f);
      const bool ray_inside_bottom_cap = \
         (edbdmath::quadraticEval(a, b, c, u_min) <= 0.f);

      // This logic tells you if the cylinder is intersected by the *line*
      // passing through ray_start and ray_end.
      geometry::types::raycastResult_t line_result;
      line_result.numHits = 0;
      if (quad_result.numRealSolutions == 0)
      {
         unsigned int & num_intersections = line_result.numHits;
         num_intersections = 0;
         if (ray_inside_top_cap)
         {
            line_result.hits[num_intersections] = u_max * ray_slope + ray_start;
            ++num_intersections;
         }
         if (ray_inside_bottom_cap)
         {
            line_result.hits[num_intersections] = u_min * ray_slope + ray_start;
            ++num_intersections;
         }

         line_result.hit = (num_intersections > 0);
      }
      else if (quad_result.numRealSolutions == 1)
      {
         if (
            (quad_result.x[0] <= u_max) &&
            (quad_result.x[0] >= u_min)
         )
         {
            line_result.hit = true;
            line_result.hits[0] = quad_result.x[0] * ray_slope + ray_start;
            line_result.numHits = 1;
            line_result.u[0] = quad_result.x[0];
         }
         else
         {
            if (ray_inside_top_cap)
            {
               line_result.hit = true;
               line_result.hits[0] = u_max * ray_slope + ray_start;
               line_result.numHits = 1;
               line_result.u[0] = u_max;
            }
            else if (ray_inside_bottom_cap)
            {
               line_result.hit = true;
               line_result.hits[0] = u_min * ray_slope + ray_start;
               line_result.numHits = 1;
               line_result.u[0] = u_min;
            }
            else
            {
               line_result.hit = false;
            }
         }
      }
      else
      {
         unsigned int & num_hits = line_result.numHits;
         num_hits = 0;

         for (unsigned int i = 0; i < quad_result.numRealSolutions; ++i)
         {
            Vector3 intersection = quad_result.x[i] * ray_slope + ray_start;
            if (
               (quad_result.x[i] <= u_max) &&
               (quad_result.x[i] >= u_min)
            )
            {
               line_result.hits[num_hits] = intersection;
               line_result.u[num_hits] = quad_result.x[i];
               ++num_hits;
            }
            else
            {
               if (quad_result.x[i] >= u_max && ray_inside_top_cap)
               {
                  line_result.hits[num_hits] = u_max * ray_slope + ray_start;
                  line_result.u[num_hits] = u_max;
                  ++num_hits;
               }
               else if (quad_result.x[i] <= u_min && ray_inside_bottom_cap)
               {
                  line_result.hits[num_hits] = u_min * ray_slope + ray_start;
                  line_result.u[num_hits] = u_min;
                  ++num_hits;
               }
            }
         }

         line_result.hit = (num_hits > 0);
      }

      // This code cleans up the result from the line intersection by only
      // keeping the values of u >= 0.0.
      geometry::types::raycastResult_t result;
      unsigned int & num_hits = result.numHits;
      num_hits = 0;

      for (unsigned int i = 0; i < line_result.numHits; ++i)
      {
         if (line_result.u[i] >= 0.f)
         {
            result.u[num_hits] = line_result.u[i];
            result.hits[num_hits] = line_result.hits[i];
            ++num_hits;
         }
      }

      result.hit = (num_hits > 0);

      return result;
   }

   // The capsule uses the cylinder raycast and sphere raycast to determine the
   // ray's hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCapsule_t & capsule_C
   )
   {
      const Vector3 ray_slope(ray_end - ray_start);
      const Vector3 z_hat_C(0.f, 0.f, 1.f);
      const Vector3 cap_bottom_W(
         geometry::transform::forwardBound(
            trans_C_to_W, -1.f * (capsule_C.height / 2) * z_hat_C
         )
      );
      const Vector3 cap_top_W(
         geometry::transform::forwardBound(
            trans_C_to_W, (capsule_C.height / 2) * z_hat_C
         )
      );

      geometry::types::shapeCylinder_t cylinder_C;
      cylinder_C.height = capsule_C.height;
      cylinder_C.radius = capsule_C.radius;

      geometry::types::shapeSphere_t sphere_C;
      sphere_C.radius = capsule_C.radius;

      geometry::types::raycastResult_t cylinder_result = raycast(
         ray_start, ray_end, trans_C_to_W, cylinder_C
      );

      geometry::types::transform_t top_sphere_C_to_W = trans_C_to_W;
      top_sphere_C_to_W.translate = cap_top_W;

      geometry::types::raycastResult_t top_sphere_result = raycast(
         ray_start, ray_end, top_sphere_C_to_W, sphere_C
      );

      geometry::types::transform_t bottom_sphere_C_to_W = trans_C_to_W;
      bottom_sphere_C_to_W.translate = cap_bottom_W;

      geometry::types::raycastResult_t bottom_sphere_result = raycast(
         ray_start, ray_end, bottom_sphere_C_to_W, sphere_C
      );

      unsigned int num_u_vals = 0;
      float u_vals[6] = {
         __FLT_MAX__,
         __FLT_MAX__,
         __FLT_MAX__,
         __FLT_MAX__,
         __FLT_MAX__,
         __FLT_MAX__
      };

      for (unsigned int i = 0; i < 2; ++i)
      {
         if (i < cylinder_result.numHits)
         {
            u_vals[num_u_vals] = cylinder_result.u[i];
            ++num_u_vals;
         }
         if (i < top_sphere_result.numHits)
         {
            u_vals[num_u_vals] = top_sphere_result.u[i];
            ++num_u_vals;
         }
         if (i < bottom_sphere_result.numHits)
         {
            u_vals[num_u_vals] = bottom_sphere_result.u[i];
            ++num_u_vals;
         }
      }

      geometry::types::raycastResult_t result;
      result.numHits = std::min(
         static_cast<unsigned int>(2), num_u_vals
      );

      if (result.numHits > 0)
      {
         std::sort(std::begin(u_vals), std::end(u_vals));

         result.u[0] = u_vals[0];
         result.u[1] = u_vals[num_u_vals - 1];

         result.hits[0] = result.u[0] * ray_slope + ray_start;
         result.hits[1] = result.u[1] * ray_slope + ray_start;
      }

      result.hit = result.numHits > 0;

      return result;
   }

   // For this case, the equation of a cube is:
   //
   // g(p_C) = (abs(p_C[0]) <= length / 2) &&
   //          (abs(p_C[1]) <= width / 2) &&
   //          (abs(p_C[2]) <= height / 2)
   //
   // This is used to find candidate values of 'u' that satisfy g(p_C(u)).
   // Each candidate pair of u_min, u_max is used to generate
   //
   //    p_C(u) = q_C + u * y_C
   //
   // Values of 'u' that satisfy g(p_C(u)) are used to calculate hit points
   // on the cuboid.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCube_t & cube_C
   )
   {
      const Vector3 ray_slope(ray_end - ray_start);
      const Matrix33 A_inv(
         (~trans_C_to_W.scale) * trans_C_to_W.rotate.transpose()
      );
      const Vector3 q_C(A_inv * (ray_start - trans_C_to_W.translate));
      const Vector3 y_C(A_inv * ray_slope);

      const Vector3 dims(
         cube_C.length / 2.f, cube_C.width / 2.f, cube_C.height / 2.f
      );

      Vector3 u_mins(
         (-1.f * dims[0] - q_C[0]) / y_C[0],
         (-1.f * dims[1] - q_C[1]) / y_C[1],
         (-1.f * dims[2] - q_C[2]) / y_C[2]
      );

      Vector3 u_maxes(
         (dims[0] - q_C[0]) / y_C[0],
         (dims[1] - q_C[1]) / y_C[1],
         (dims[2] - q_C[2]) / y_C[2]
      );

      geometry::types::raycastResult_t result;
      unsigned int & num_hits = result.numHits;
      num_hits = 0;

      for (unsigned int i = 0; i < 3; ++i)
      {
         float u_min = u_mins[i];
         float u_max = u_maxes[i];

         // A small margin of 1e-5f is used to pad the dimensions, otherwise
         // very long cuboids might not register ray intersections.
         const Vector3 intersect_min_C(u_min * y_C + q_C);
         if (
            (std::fabs(intersect_min_C[0]) <= dims[0] + 1e-5f) &&
            (std::fabs(intersect_min_C[1]) <= dims[1] + 1e-5f) &&
            (std::fabs(intersect_min_C[2]) <= dims[2] + 1e-5f) &&
            u_min >= 0.f
         )
         {
            result.hits[num_hits] = u_min * ray_slope + ray_start;
            result.u[num_hits] = u_min;
            ++num_hits;
         }

         if (num_hits > 1)
         {
            break;
         }

         const Vector3 intersect_max_C(u_max * y_C + q_C);
         if (
            (std::fabs(intersect_max_C[0]) <= dims[0] + 1e-5f) &&
            (std::fabs(intersect_max_C[1]) <= dims[1] + 1e-5f) &&
            (std::fabs(intersect_max_C[2]) <= dims[2] + 1e-5f) &&
            u_max >= 0.f
         )
         {
            result.hits[num_hits] = u_max * ray_slope + ray_start;
            result.u[num_hits] = u_max;
            ++num_hits;
         }

         if (num_hits > 1)
         {
            break;
         }
      }

      result.hit = num_hits > 0;

      return result;
   }

   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shape_t & shape_C
   )
   {
      geometry::types::raycastResult_t result;
      result.hit = false;

      switch(shape_C.shapeType)
      {
         case geometry::types::enumShape_t::CUBE:
         {
            result = raycast(ray_start, ray_end, trans_C_to_W, shape_C.cube);
            break;
         }
         case geometry::types::enumShape_t::SPHERE:
         {
            result = raycast(ray_start, ray_end, trans_C_to_W, shape_C.sphere);
            break;
         }
         case geometry::types::enumShape_t::CAPSULE:
         {
            result = raycast(ray_start, ray_end, trans_C_to_W, shape_C.capsule);
            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            result = raycast(ray_start, ray_end, trans_C_to_W, shape_C.cylinder);
            break;
         }
         default:
            break;
      }

      return result;
   }

   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeSphere_t & sphere_C
   )
   {
      const Vector3 ray_slope(ray_end - ray_start);
      const Matrix33 A_inv(trans_C_to_W.rotate.transpose());
      const Vector3 q_C(A_inv * (ray_start - trans_C_to_W.translate));
      const Vector3 y_C(A_inv * ray_slope);

      const float q_2 = q_C.magnitudeSquared();
      const float y_dot_q = y_C.dot(q_C);
      const float y_2 = y_C.magnitudeSquared();

      // a * u ^ 2 + b * u + c = 0
      edbdmath::quadraticResult_t quad_result = edbdmath::quadraticRoots(
         y_2, 2.f * y_dot_q, q_2 - sphere_C.radius * sphere_C.radius
      );

      geometry::types::raycastResult_t result;
      if (quad_result.numRealSolutions == 0)
      {
         result.numHits = 0;
         result.hit = false;
      }
      else if (quad_result.numRealSolutions == 1)
      {
         if (quad_result.x[0] >= 0.f)
         {
            result.hit = true;
            result.numHits = quad_result.numRealSolutions;
            result.hits[0] = quad_result.x[0] * ray_slope + ray_start;
            result.u[0] = quad_result.x[0];
         }
         else
         {
            result.numHits = 0;
            result.hit = false;
         }
      }
      else
      {
         unsigned int & num_hits = result.numHits;
         num_hits = 0;

         for (unsigned int i = 0; i < quad_result.numRealSolutions; ++i)
         {
            Vector3 intersection = quad_result.x[i] * ray_slope + ray_start;
            if (quad_result.x[i] >= 0.f)
            {
               result.hits[num_hits] = intersection;
               result.u[num_hits] = quad_result.x[i];
               ++num_hits;
            }
         }

         result.hit = (num_hits > 0);
      }

      return result;
   }

   // The equation for a cylinder is given as:
   //
   //    g(p_C) = (p_C ^ T) * H * p_C - (R ^ 2) = 0
   //
   //       abs(p_C[2] <= cylinder_C.height / 2
   //
   //       H = diag(1.f, 1.f, 0.f)
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCylinder_t & cylinder_C
   )
   {
      const Vector3 ray_slope(ray_end - ray_start);
      const Matrix33 A_inv(trans_C_to_W.rotate.transpose());
      const Vector3 q_C(A_inv * (ray_start - trans_C_to_W.translate));
      const Vector3 y_C(A_inv * ray_slope);
      const Matrix33 H(1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f);

      const Vector3 H_dot_y = H * y_C;
      const float a = y_C.dot(H_dot_y);
      const float b = 2.f * q_C.dot(H_dot_y);
      const float c = q_C.dot(H * q_C) - cylinder_C.radius * cylinder_C.radius;

      // a * u ^ 2 + b * u + c = 0
      // The solution is the intersection between the line through
      // [ray_start, ray_end] and the infinite cylinder.
      edbdmath::quadraticResult_t quad_result = edbdmath::quadraticRoots(a, b, c);

      // The maximum and minimum bounds on u based on the cylinder's height.
      const float u_bound1 = (-1.f * cylinder_C.height / 2 - q_C[2]) / (y_C[2]);
      const float u_bound2 = (cylinder_C.height / 2 - q_C[2]) / (y_C[2]);

      // Min and max values of u based on the bounds from the cap.
      const float u_min = std::min(u_bound1, u_bound2);
      const float u_max = std::max(u_bound1, u_bound2);

      const bool ray_inside_top_cap = \
         (edbdmath::quadraticEval(a, b, c, u_max) <= 0.f);
      const bool ray_inside_bottom_cap = \
         (edbdmath::quadraticEval(a, b, c, u_min) <= 0.f);

      // This logic tells you if the cylinder is intersected by the *line*
      // passing through ray_start and ray_end.
      geometry::types::raycastResult_t line_result;
      line_result.numHits = 0;
      if (quad_result.numRealSolutions == 0)
      {
         unsigned int & num_intersections = line_result.numHits;
         num_intersections = 0;
         if (ray_inside_top_cap)
         {
            line_result.hits[num_intersections] = u_max * ray_slope + ray_start;
            ++num_intersections;
         }
         if (ray_inside_bottom_cap)
         {
            line_result.hits[num_intersections] = u_min * ray_slope + ray_start;
            ++num_intersections;
         }

         line_result.hit = (num_intersections > 0);
      }
      else if (quad_result.numRealSolutions == 1)
      {
         if (
            (quad_result.x[0] <= u_max) &&
            (quad_result.x[0] >= u_min)
         )
         {
            line_result.hit = true;
            line_result.hits[0] = quad_result.x[0] * ray_slope + ray_start;
            line_result.numHits = 1;
            line_result.u[0] = quad_result.x[0];
         }
         else
         {
            if (ray_inside_top_cap)
            {
               line_result.hit = true;
               line_result.hits[0] = u_max * ray_slope + ray_start;
               line_result.numHits = 1;
               line_result.u[0] = u_max;
            }
            else if (ray_inside_bottom_cap)
            {
               line_result.hit = true;
               line_result.hits[0] = u_min * ray_slope + ray_start;
               line_result.numHits = 1;
               line_result.u[0] = u_min;
            }
            else
            {
               line_result.hit = false;
            }
         }
      }
      else
      {
         unsigned int & num_hits = line_result.numHits;
         num_hits = 0;

         for (unsigned int i = 0; i < quad_result.numRealSolutions; ++i)
         {
            Vector3 intersection = quad_result.x[i] * ray_slope + ray_start;
            if (
               (quad_result.x[i] <= u_max) &&
               (quad_result.x[i] >= u_min)
            )
            {
               line_result.hits[num_hits] = intersection;
               line_result.u[num_hits] = quad_result.x[i];
               ++num_hits;
            }
            else
            {
               if (quad_result.x[i] >= u_max && ray_inside_top_cap)
               {
                  line_result.hits[num_hits] = u_max * ray_slope + ray_start;
                  line_result.u[num_hits] = u_max;
                  ++num_hits;
               }
               else if (quad_result.x[i] <= u_min && ray_inside_bottom_cap)
               {
                  line_result.hits[num_hits] = u_min * ray_slope + ray_start;
                  line_result.u[num_hits] = u_min;
                  ++num_hits;
               }
            }
         }

         line_result.hit = (num_hits > 0);
      }

      // This code cleans up the result from the line intersection by only
      // keeping the values of u >= 0.0.
      geometry::types::raycastResult_t result;
      unsigned int & num_hits = result.numHits;
      num_hits = 0;

      for (unsigned int i = 0; i < line_result.numHits; ++i)
      {
         if (line_result.u[i] >= 0.f)
         {
            result.u[num_hits] = line_result.u[i];
            result.hits[num_hits] = line_result.hits[i];
            ++num_hits;
         }
      }

      result.hit = (num_hits > 0);

      return result;
   }

   // The capsule uses the cylinder raycast and sphere raycast to determine the
   // ray's hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCapsule_t & capsule_C
   )
   {
      const Vector3 ray_slope(ray_end - ray_start);
      const Vector3 z_hat_C(0.f, 0.f, 1.f);
      const Vector3 cap_bottom_W(
         geometry::transform::forwardBound(
            trans_C_to_W, -1.f * (capsule_C.height / 2) * z_hat_C
         )
      );
      const Vector3 cap_top_W(
         geometry::transform::forwardBound(
            trans_C_to_W, (capsule_C.height / 2) * z_hat_C
         )
      );

      geometry::types::shapeCylinder_t cylinder_C;
      cylinder_C.height = capsule_C.height;
      cylinder_C.radius = capsule_C.radius;

      geometry::types::shapeSphere_t sphere_C;
      sphere_C.radius = capsule_C.radius;

      geometry::types::raycastResult_t cylinder_result = raycast(
         ray_start, ray_end, trans_C_to_W, cylinder_C
      );

      geometry::types::isometricTransform_t temp_sphere_C_to_W = trans_C_to_W;
      temp_sphere_C_to_W.translate = cap_top_W;

      geometry::types::raycastResult_t top_sphere_result = raycast(
         ray_start, ray_end, temp_sphere_C_to_W, sphere_C
      );

      temp_sphere_C_to_W.translate = cap_bottom_W;

      geometry::types::raycastResult_t bottom_sphere_result = raycast(
         ray_start, ray_end, temp_sphere_C_to_W, sphere_C
      );

      unsigned int num_u_vals = 0;
      float u_vals[6] = {
         __FLT_MAX__,
         __FLT_MAX__,
         __FLT_MAX__,
         __FLT_MAX__,
         __FLT_MAX__,
         __FLT_MAX__
      };

      for (unsigned int i = 0; i < 2; ++i)
      {
         if (i < cylinder_result.numHits)
         {
            u_vals[num_u_vals] = cylinder_result.u[i];
            ++num_u_vals;
         }
         if (i < top_sphere_result.numHits)
         {
            u_vals[num_u_vals] = top_sphere_result.u[i];
            ++num_u_vals;
         }
         if (i < bottom_sphere_result.numHits)
         {
            u_vals[num_u_vals] = bottom_sphere_result.u[i];
            ++num_u_vals;
         }
      }

      geometry::types::raycastResult_t result;
      result.numHits = std::min(
         static_cast<unsigned int>(2), num_u_vals
      );

      if (result.numHits > 0)
      {
         std::sort(std::begin(u_vals), std::end(u_vals));

         result.u[0] = u_vals[0];
         result.u[1] = u_vals[num_u_vals - 1];

         result.hits[0] = result.u[0] * ray_slope + ray_start;
         result.hits[1] = result.u[1] * ray_slope + ray_start;
      }

      result.hit = result.numHits > 0;

      return result;
   }

   // For this case, the equation of a cube is:
   //
   // g(p_C) = (abs(p_C[0]) <= length / 2) &&
   //          (abs(p_C[1]) <= width / 2) &&
   //          (abs(p_C[2]) <= height / 2)
   //
   // This is used to find candidate values of 'u' that satisfy g(p_C(u)).
   // Each candidate pair of u_min, u_max is used to generate
   //
   //    p_C(u) = q_C + u * y_C
   //
   // Values of 'u' that satisfy g(p_C(u)) are used to calculate hit points
   // on the cuboid.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCube_t & cube_C
   )
   {
      const Vector3 ray_slope(ray_end - ray_start);
      const Matrix33 A_inv(trans_C_to_W.rotate.transpose());
      const Vector3 q_C(A_inv * (ray_start - trans_C_to_W.translate));
      const Vector3 y_C(A_inv * ray_slope);

      const Vector3 dims(
         cube_C.length / 2.f, cube_C.width / 2.f, cube_C.height / 2.f
      );

      Vector3 u_mins(
         (-1.f * dims[0] - q_C[0]) / y_C[0],
         (-1.f * dims[1] - q_C[1]) / y_C[1],
         (-1.f * dims[2] - q_C[2]) / y_C[2]
      );

      Vector3 u_maxes(
         (dims[0] - q_C[0]) / y_C[0],
         (dims[1] - q_C[1]) / y_C[1],
         (dims[2] - q_C[2]) / y_C[2]
      );

      geometry::types::raycastResult_t result;
      unsigned int & num_hits = result.numHits;
      num_hits = 0;

      for (unsigned int i = 0; i < 3; ++i)
      {
         float u_min = u_mins[i];
         float u_max = u_maxes[i];

         // A small margin of 1e-5f is used to pad the dimensions, otherwise
         // very long cuboids might not register ray intersections.
         const Vector3 intersect_min_C(u_min * y_C + q_C);
         if (
            (std::fabs(intersect_min_C[0]) <= dims[0] + 1e-5f) &&
            (std::fabs(intersect_min_C[1]) <= dims[1] + 1e-5f) &&
            (std::fabs(intersect_min_C[2]) <= dims[2] + 1e-5f) &&
            u_min >= 0.f
         )
         {
            result.hits[num_hits] = u_min * ray_slope + ray_start;
            result.u[num_hits] = u_min;
            ++num_hits;
         }

         if (num_hits > 1)
         {
            break;
         }

         const Vector3 intersect_max_C(u_max * y_C + q_C);
         if (
            (std::fabs(intersect_max_C[0]) <= dims[0] + 1e-5f) &&
            (std::fabs(intersect_max_C[1]) <= dims[1] + 1e-5f) &&
            (std::fabs(intersect_max_C[2]) <= dims[2] + 1e-5f) &&
            u_max >= 0.f
         )
         {
            result.hits[num_hits] = u_max * ray_slope + ray_start;
            result.u[num_hits] = u_max;
            ++num_hits;
         }

         if (num_hits > 1)
         {
            break;
         }
      }

      result.hit = num_hits > 0;

      return result;
   }

   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shape_t & shape_C
   )
   {
      geometry::types::raycastResult_t result;
      result.hit = false;

      switch(shape_C.shapeType)
      {
         case geometry::types::enumShape_t::CUBE:
         {
            result = raycast(ray_start, ray_end, trans_C_to_W, shape_C.cube);
            break;
         }
         case geometry::types::enumShape_t::SPHERE:
         {
            result = raycast(ray_start, ray_end, trans_C_to_W, shape_C.sphere);
            break;
         }
         case geometry::types::enumShape_t::CAPSULE:
         {
            result = raycast(ray_start, ray_end, trans_C_to_W, shape_C.capsule);
            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            result = raycast(ray_start, ray_end, trans_C_to_W, shape_C.cylinder);
            break;
         }
         default:
            break;
      }

      return result;
   }

}
