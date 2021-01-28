#include "segment.hpp"

#include "line.hpp"
#include "plane.hpp"

#include <cmath>

namespace geometry
{

namespace segment
{
   float closestParameterToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & q
   )
   {
      Vector3 bma = b - a;
      float t = (q - a).dot(bma)/bma.magnitudeSquared();

      t = std::fmax(
         std::fmin(t, 1.0f),
         0.0f
      );

      return t;
   }

   pointBaryCoord_t closestPointToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & q
   )
   {
      float t = closestParameterToPoint(a, b, q);

      pointBaryCoord_t closest_point;
      closest_point.point = (1.0f - t) * a + t * b;
      closest_point.bary[0] = 1.0f - t;
      closest_point.bary[1] = t;
      closest_point.bary[2] = 0.0f;
      closest_point.bary[3] = 0.0f;

      return closest_point;
   }

   geometry::types::segmentClosestPoints_t closestPointsToSegment(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      float epsilon
   )
   {
      geometry::types::segmentClosestPoints_t result;

      epsilon = std::fmin(std::fmax(epsilon, 0.f), 1e-1f);

      float seg_a_len2 = (a - b).magnitudeSquared();
      float seg_b_len2 = (c - d).magnitudeSquared();

      if (seg_a_len2 < 1e-7f && seg_b_len2 < 1e-7f)
      {
         result.segmentPoints[0] = a;
         result.otherPoints[0] = c;
         result.numPairs = 1;
         return result;
      }

      if (seg_a_len2 < 1e-7f)
      {
         result.segmentPoints[0] = a;
         result.otherPoints[0] = closestPointToPoint(c, d, a).point;
         result.numPairs = 1;
         return result;
      }

      if (seg_b_len2 < 1e-7f)
      {
         result.segmentPoints[0] = closestPointToPoint(a, b, c).point;
         result.otherPoints[0] = c;
         result.numPairs = 1;
         return result;
      }

      const Vector3 e = b - a;
      const Vector3 f = d - c;
      const float e_2 = e.magnitudeSquared();

      const float u = e_2;
      const float v = e.dot(f);
      const float w = f.magnitudeSquared();

      const float denom = u * w - v * v;

      int num_line_closest_points = (denom < 1e-7f) ? 2 : 1;

      // Non-degenerate case (segments aren't parallel)
      if (num_line_closest_points == 1)
      {
         // Follows this progression:
         // 1. Find the parameter 'i' of p(i) = a + (b - a) * i for the line
         //    l_P that's closest to line l_Q, where q(t) = c + (d - c) * t
         // 2. Clamp 'i' to the range [0, 1]
         // 3. Find the parameter 't' of the line l_Q that's closest to the
         //    point p(i), where 'i' from step 2 is used
         // 4. If 't' is outside the range [0, 1], clamp 't', then find the
         //    parameter 'i' on the line l_P that's closest to q(t), where 't'
         //    is either 0 or 1
         const float y = e.dot(c - a);
         const float z = f.dot(a - c);

         float i = (y * w + z * v) / denom;
         i = std::min(std::max(i, 0.f), 1.f);
         
         float t = (f.dot(a + e * i - c)) / w;

         if (t < 0.f)
         {
            t = 0.f;
            i = e.dot(c - a) / u;
            i = std::min(std::max(i, 0.f), 1.f);
         }
         else if (t > 1.f)
         {
            t = 1.f;
            i = e.dot(d - a) / u;
            i = std::min(std::max(i, 0.f), 1.f);
         }

         result.segmentPoints[0] = a + e * i;
         result.otherPoints[0] = c + f * t;

         result.numPairs = 1;
         return result;
      }

      // Ok, now the degenerate case.
      // Since p(i) and q(t) are parallel, find the values of 'i' and 't'
      // where p(i) and q(t) have a continuum of closest points.
      //
      // 1. Make p(i) and q(t) completely colinear by moving q(t) onto the
      //    line l_P, creating the segment [c2, d2]
      // 2. Find the parameters of 'i' that place c2 on p(i) and d2 on p(i)
      // 3. Clamp those parameters to the range [0, 1]
      // 4. If the parameters are the same, then there's only one pair of
      //    closest points, return them
      // 5. Return the bounding pair of closest points

      // Force [c, d] to be parallel to [a, b] by aligning f with e.
      const Vector3 e_unit = e.unitVector();
      const Vector3 c1 = c;
      const Vector3 f1 = f.dot(e_unit) * e_unit;

      // Make [c1, d1] colinear with [a, b] by shifting c1 and d1 to [c2, d2].
      const Vector3 n = geometry::line::closestPointToPoint(a, b, c1);
      const Vector3 c2 = n;
      const Vector3 d2 = n + f1;

      // My derivation used 's' as the parameterization for
      //    r1(s) = a + (b - a) * s
      // Unfortunate naming convention, where this function returns 's' as one
      // of the closest points.
      float s_c2 = (c2 - a).dot(e) / e_2;
      float s_d2 = (d2 - a).dot(e) / e_2;

      float s0 = std::fmin(std::fmax(s_c2, 0.f), 1.f);
      float s1 = std::fmin(std::fmax(s_d2, 0.f), 1.f);

      if (fabs(s0 - s1) < 1e-7f)
      {
         result.segmentPoints[0] = a + e * std::fmin(s0, s1);
         result.otherPoints[0] = closestPointToPoint(
            c, d, result.segmentPoints[0]
         ).point;
         result.numPairs = 1;
         return result;
      }

      result.segmentPoints[0] = a + e * s0;
      result.segmentPoints[1] = a + e * s1;

      result.otherPoints[0] = closestPointToPoint(
         c, d, result.segmentPoints[0]
      ).point;
      result.otherPoints[1] = closestPointToPoint(
         c, d, result.segmentPoints[1]
      ).point;

      result.numPairs = 2;
      return result;
   }

   geometry::types::segmentClosestPoints_t closestPointsToPlane(
      const Vector3 & a,
      const Vector3 & b,
      const geometry::types::plane_t & plane
   )
   {
      const Vector3 m = b - a;
      const Vector3 m_hat = m.unitVector();
      const Vector3 n_hat = plane.normal.unitVector();

      // Segment is parallel to the plane
      geometry::types::segmentClosestPoints_t result;
      if (std::fabs(m_hat.dot(n_hat)) <= 1e-7f)
      {
         result.numPairs = 2;
         result.segmentPoints[0] = a;
         result.segmentPoints[1] = b;
         result.otherPoints[0] = geometry::plane::closestPointToPoint(
            plane.normal, plane.point, a
         );
         result.otherPoints[1] = geometry::plane::closestPointToPoint(
            plane.normal, plane.point, b
         );

         return result;
      }

      result.numPairs = 1;
      const float signed_distance_a = plane.normal.dot(a - plane.point);
      const float signed_distance_b = plane.normal.dot(b - plane.point);

      // End points are on the same side of the plane
      if (std::signbit(signed_distance_a) == std::signbit(signed_distance_b))
      {
         result.segmentPoints[0] = (
            (std::fabs(signed_distance_a) <= std::fabs(signed_distance_b))
            ? a
            : b
         );

         result.otherPoints[0] = geometry::plane::closestPointToPoint(
            plane.normal, plane.point, result.segmentPoints[0]
         );

         return result;
      }

      const float t = (
         plane.normal.dot(plane.point - a) / plane.normal.dot(b - a)
      );

      result.numPairs = 1;
      result.segmentPoints[0] = a + (b - a) * t;
      result.otherPoints[0] = result.segmentPoints[0];

      return result;
   }

   geometry::types::segmentClosestPoints_t closestPointsToAabb(
      const Vector3 & a,
      const Vector3 & b,
      const geometry::types::aabb_t & aabb
   )
   {
      // The closest points between the segment and the AABB
      float closest_dist2 = __FLT_MAX__;
      geometry::types::segmentClosestPoints_t closest_points;

      const Vector3 u = aabb.vertMin;
      const Vector3 v = aabb.vertMax;

      // Handle the degenerate case.
      if ((a - b).magnitude() < 1e-7f)
      {
         closest_points.numPairs = 1;
         closest_points.segmentPoints[0] = a;
         for (int i = 0; i < 3; ++i)
         {
            closest_points.otherPoints[0][i] = std::min(
               std::max(a[i], u[i]),
               v[i]
            );
         }

         return closest_points;
      }

      const int num_edges = 12;
      const geometry::types::segment_t edge_segments[num_edges] = {
         // x-hat direction
         {Vector3(u[0], u[1], u[2]), Vector3(v[0], u[1], u[2])},
         {Vector3(u[0], u[1], v[2]), Vector3(v[0], u[1], v[2])},
         {Vector3(u[0], v[1], u[2]), Vector3(v[0], v[1], u[2])},
         {Vector3(u[0], v[1], v[2]), Vector3(v[0], v[1], v[2])},
         // y-hat direction
         {Vector3(u[0], u[1], u[2]), Vector3(u[0], v[1], u[2])},
         {Vector3(u[0], u[1], v[2]), Vector3(u[0], v[1], v[2])},
         {Vector3(v[0], u[1], u[2]), Vector3(v[0], v[1], u[2])},
         {Vector3(v[0], u[1], v[2]), Vector3(v[0], v[1], v[2])},
         // z-hat direction
         {Vector3(u[0], u[1], u[2]), Vector3(u[0], u[1], v[2])},
         {Vector3(u[0], v[1], u[2]), Vector3(u[0], v[1], v[2])},
         {Vector3(v[0], u[1], u[2]), Vector3(v[0], u[1], v[2])},
         {Vector3(v[0], v[1], u[2]), Vector3(v[0], v[1], v[2])},
      };

      // Find the closest points between the query segment and the AABB edge
      // segments.
      for (int i = 0; i < num_edges; ++i)
      {
         auto temp_edge_closest_points = closestPointsToSegment(
            a, b, edge_segments[i].points[0], edge_segments[i].points[1], 1e-6f
         );

         const float temp_dist2 = (
            temp_edge_closest_points.segmentPoints[0] - temp_edge_closest_points.otherPoints[0]
         ).magnitudeSquared();

         if (temp_dist2 < closest_dist2)
         {
            closest_dist2 = temp_dist2;
            closest_points = temp_edge_closest_points;
         }

         // Found an intersection between the edge and the segment, this the
         // closest two segments can get to each other.
         if (temp_dist2 < 1e-7f)
         {
            break;
         }
      }

      const Vector3 aabb_center = (u + v) / 2.f;
      const int num_faces = 6;

      // Find the valid closest points between the segment and AABB faces.
      const geometry::types::plane_t faces[num_faces] = {
         // x-hat
         {Vector3(u[0], aabb_center[1], aabb_center[2]), Vector3(-1.f, 0.f, 0.f)},
         {Vector3(v[0], aabb_center[1], aabb_center[2]), Vector3( 1.f, 0.f, 0.f)},
         // y-hat
         {Vector3(aabb_center[0], u[1], aabb_center[2]), Vector3(0.f, -1.f, 0.f)},
         {Vector3(aabb_center[0], v[1], aabb_center[2]), Vector3(0.f,  1.f, 0.f)},
         // z-hat
         {Vector3(aabb_center[0], aabb_center[1], u[2]), Vector3(0.f, 0.f, -1.f)},
         {Vector3(aabb_center[0], aabb_center[1], v[2]), Vector3(0.f, 0.f,  1.f)}
      };

      for (int i = 0; i < num_faces; ++i)
      {
         const auto temp_face_points = closestPointsToPlane(a, b, faces[i]);

         if (temp_face_points.numPairs == 1)
         {
            float temp_dist2 = (
               temp_face_points.otherPoints[0] - temp_face_points.segmentPoints[0]
            ).magnitudeSquared();

            const int face_index = i / 2;
            const Vector3 & face_point = temp_face_points.otherPoints[0];

            if (
               (face_point[(face_index + 1) % 3] < u[(face_index + 1) % 3]) ||
               (face_point[(face_index + 1) % 3] > v[(face_index + 1) % 3]) ||
               (face_point[(face_index + 2) % 3] < u[(face_index + 2) % 3]) ||
               (face_point[(face_index + 2) % 3] > v[(face_index + 2) % 3])
            )
            {
               continue;
            }

            if (temp_dist2 < closest_dist2)
            {
               closest_dist2 = temp_dist2;
               closest_points = temp_face_points;
            }

            if (temp_dist2 < 1e-7f)
            {
               break;
            }
         }
         else if (temp_face_points.numPairs == 2)
         {
            // Clip the segment using the planes that bound this face... ugh.
            geometry::types::clippedSegment_t clipped_segment = \
               clipSegmentAgainstAxisAlignedSquare(
                  temp_face_points.segmentPoints[0],
                  temp_face_points.segmentPoints[1],
                  aabb,
                  i / 2
               );

            if (!clipped_segment.clipped)
            {
               continue;
            }

            geometry::types::clippedSegment_t clipped_face_segment = \
               clipSegmentAgainstAxisAlignedSquare(
                  temp_face_points.otherPoints[0],
                  temp_face_points.otherPoints[1],
                  aabb,
                  i / 2
               );

            const int face_index = i / 2;
            const Vector3 & face_point0 = clipped_face_segment.points[0];
            const Vector3 & face_point1 = clipped_face_segment.points[1];

            if (
               (face_point0[(face_index + 1) % 3] < u[(face_index + 1) % 3]) ||
               (face_point0[(face_index + 1) % 3] > v[(face_index + 1) % 3]) ||
               (face_point0[(face_index + 2) % 3] < u[(face_index + 2) % 3]) ||
               (face_point0[(face_index + 2) % 3] > v[(face_index + 2) % 3]) ||
               (face_point1[(face_index + 1) % 3] < u[(face_index + 1) % 3]) ||
               (face_point1[(face_index + 1) % 3] > v[(face_index + 1) % 3]) ||
               (face_point1[(face_index + 2) % 3] < u[(face_index + 2) % 3]) ||
               (face_point1[(face_index + 2) % 3] > v[(face_index + 2) % 3])
            )
            {
               continue;
            }

            float temp_dist2 = (
               clipped_segment.points[0] - clipped_face_segment.points[0]
            ).magnitudeSquared();

            if (temp_dist2 <= closest_dist2)
            {
               closest_dist2 = temp_dist2;
               closest_points.numPairs = 2;
               closest_points.segmentPoints[0] = clipped_segment.points[0];
               closest_points.segmentPoints[1] = clipped_segment.points[1];
               closest_points.otherPoints[0] = clipped_face_segment.points[0];
               closest_points.otherPoints[1] = clipped_face_segment.points[1];
            }

            if (temp_dist2 < 1e-7f)
            {
               break;
            }
         }
      }

      return closest_points;
   }

   geometry::types::clippedSegment_t clipSegmentAgainstAxisAlignedSquare(
      const Vector3 & a,
      const Vector3 & b,
      const geometry::types::aabb_t & aabb,
      const int face_normal_index
   )
   {
      const Vector3 & u = aabb.vertMin;
      const Vector3 & v = aabb.vertMax;
      const Vector3 aabb_center = (u + v) / 2.f;

      const int i = face_normal_index;

      geometry::types::clippedSegment_t clipped_segment;
      clipped_segment.clipped = true;
      clipped_segment.points[0] = a;
      clipped_segment.points[1] = b;

      // Indicates that some part of the original segment was interior to the
      // area defined by the axis-aligned bounding planes.
      bool clipped = false;

      for (int j = 0; j < 4; ++j)
      {
         const int k = ((j % 2) + i) % 3;
         geometry::types::plane_t clip_plane;
         clip_plane.normal[k] = (j < 2) ? -1.f : 1.f;
         clip_plane.point = aabb_center;
         clip_plane.point[k] = (j < 2) ? v[k] : u[k];

         const auto temp_clipped_segment = geometry::plane::clipSegment(
            clipped_segment.points[0], clipped_segment.points[1], clip_plane
         );

         clipped |= temp_clipped_segment.clipped;

         if (!temp_clipped_segment.clipped)
         {
            continue;
         }

         clipped_segment = temp_clipped_segment;
      }

      clipped_segment.clipped = clipped;
      return clipped_segment;
   }

   bool pointColinear(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & q,
      float epsilon
   )
   {
      Vector3 p = closestPointToPoint(a, b, q).point;
      return (q - p).magnitudeSquared() <= epsilon;
   }

   int intersection(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      Vector3 & p,
      Vector3 & q,
      float epsilon
   )
   {
      Vector3 e = b - a;
      Vector3 f = d - c;

      // Check for segments that are actually points
      if (f.magnitude() < epsilon)
      {
         Vector3 closest_pt = closestPointToPoint(a, b, c).point;
         if ((closest_pt - c).magnitude() < epsilon)
         {
            p = c;
            return 1;
         }

         return 0;
      }

      if (e.magnitude() < epsilon)
      {
         Vector3 closest_pt = closestPointToPoint(c, d, a).point;
         if ((closest_pt - a).magnitude() < epsilon)
         {
            p = a;
            return 1;
         }

         return 0;
      }

      if ((e.magnitude() < epsilon) && (f.magnitude() < epsilon))
      {
         if ((a - c).magnitude() < epsilon)
         {
            p = a;
            return 1;
         }

         return 0;
      }

      Vector3 e_x_f = e.crossProduct(f);
      float mag_e_x_f_sq = e_x_f.magnitudeSquared();
      float mag_e = e.magnitude();
      float mag_f = f.magnitude();

      // Zero or infinite intersections if e x f ~= 0.
      //    Zero intersections if bounds on u are not respected
      //       -(c - a).dot(e) >= mag(e) * mag(f)
      //       or (c - a).dot(e) >= mag(e) * mag(e)
      //    Infinite intersections if bounds on u are respected

      if (mag_e_x_f_sq <= epsilon)
      {
         float t_c = (c - a).dot(e) / (mag_e * mag_e);
         float t_d = (d - a).dot(e) / (mag_e * mag_e);

         float u_a = (a - c).dot(f) / (mag_f * mag_f);
         float u_b = (b - c).dot(f) / (mag_f * mag_f);

         float mag_c_m_a_x_f_sq = (c - a).crossProduct(f).magnitudeSquared();

         if (
            (
               (t_c > 1.f || t_c < 0.f) &&
               (t_d > 1.f || t_d < 0.f) &&
               (u_a > 1.f || u_a < 0.f) &&
               (u_b > 1.f || u_b < 0.f)
            ) ||
            (mag_c_m_a_x_f_sq > epsilon)
         )
         {
            return 0;
         }

         float t_values[4] = {
            0.f, 1.f, t_c, t_d
         };

         std::sort(&t_values[0], &t_values[0] + 4);

         p = a + t_values[1] * e;
         q = a + t_values[2] * e;

         return 2;
      }

      // One intersection point.
      float t = ((c - a).crossProduct(f)).dot(e_x_f) / mag_e_x_f_sq;
      float u = ((c - a).crossProduct(e)).dot(e_x_f) / mag_e_x_f_sq;
      if (mag_e_x_f_sq > 1e-14 && t >= 0.f && u >= 0.f && t <= 1.f && u <= 1.f)
      {
         p = a + t * e;
         q = c + u * f;
         return 1;
      }

      // No intersection points.
      return 0;
   }

   float length(const Vector3 & a, const Vector3 & b)
   {
      return (a - b).magnitude();
   }

}

}
