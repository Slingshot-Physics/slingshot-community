#include "line.hpp"

#include <cmath>

namespace geometry
{

namespace line
{

   float closestParameterToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & q
   )
   {
      Vector3 bma = b - a;
      float t = (q - a).dot(bma)/bma.magnitudeSquared();

      return t;
   }

   // Find the point on the segment between a and b closest to point c.
   Vector3 closestPointToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & q
   )
   {
      Vector3 closest_point;
      float t = closestParameterToPoint(a, b, q);
      closest_point = t * b + (1.0f - t) * a;

      return closest_point;
   }

   int closestPointsToLine(
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

      epsilon = std::fmin(std::fmax(epsilon, 1e-7f), 0.1f);

      if (e.unitVector().dot(f.unitVector()) >= 1.f - epsilon)
      {
         p = a;
         q = c;
         return 2;
      }

      float e_dot_f = e.dot(f);
      float f_2 = f.dot(f);
      float e_2 = e.dot(e);
      float det_A = e_dot_f * e_dot_f - e_2 * f_2;
      Vector3 c_m_a = c - a;

      // Parameterizes the line r(s) = a + (b - a) * s
      float s = 0.f;

      // Parameterizes the line r(t) = c + (d - c) * t
      float t = 0.f;

      s = (e_dot_f) * f.dot(c_m_a) - 1.f * f_2 * e.dot(c_m_a);
      s /= det_A;

      t = e_2 * f.dot(c_m_a) - e_dot_f * e.dot(c_m_a);
      t /= det_A;

      p = a + (b - a) * s;
      q = c + (d - c) * t;
      return 1;
   }

   bool pointColinear(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & q,
      float epsilon
   )
   {
      Vector3 closest_point = closestPointToPoint(a, b, q);
      return (closest_point - q).magnitudeSquared() <= epsilon;
   }

   int segmentIntersection(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      Vector3 & p,
      Vector3 & q
   )
   {
      Vector3 e = b - a;
      Vector3 f = d - c;
      Vector3 e_x_f = e.crossProduct(f);
      float mag_e_x_f_sq = e_x_f.magnitudeSquared();
      float mag_c_m_a_x_e = (c - a).crossProduct(e).magnitude();

      if (mag_e_x_f_sq <= 1e-11)
      {
         if (mag_c_m_a_x_e <= 1e-7)
         {
            p = c;
            q = d;
            return 2;
         }

         return 0;
      }

      // One intersection point.
      float t = (((c - a).crossProduct(f)).dot(e_x_f)) / mag_e_x_f_sq;
      float u = (((c - a).crossProduct(e)).dot(e_x_f)) / mag_e_x_f_sq;
      if (u >= 0.f && u <= 1.f)
      {
         p = a + t * e;
         q = c + u * f;

         // Verified with tool that float-based intersections can cause
         // intersection points to be separated by distances between 1e-5 and
         // 0.0.
         if ((p - q).magnitude() < 1e-5f)
         {
            return 1;
         }
      }

      return 0;
   }

} // line

} // geometry
