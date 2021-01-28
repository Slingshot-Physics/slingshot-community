#include "plane.hpp"

#include <cmath>

namespace geometry
{

namespace plane
{

   Vector3 closestPointToPoint(
      const Vector3 & n, const Vector3 & a, const Vector3 & q
   )
   {
      float t = n.dot(q - a)/n.magnitudeSquared();
      Vector3 p = q - t * n;
      return p;
   }

   Vector3 closestPointToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & c, const Vector3 & q
   )
   {
      Vector3 n = (a - b).crossProduct(c - b);
      float t = n.dot(q - a)/n.magnitudeSquared();
      Vector3 p = q - t * n;
      return p;
   }

   bool pointCoplanar(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q,
      float epsilon
   )
   {
      return (q - closestPointToPoint(a, b, c, q)).magnitude() <= epsilon;
   }

   bool rayIntersect(   
      const Vector3 & normal,
      const Vector3 & a,
      const Vector3 & ray_start,
      const Vector3 & ray_unit,
      Vector3 & intersection
   )
   {
      Vector3 normal_unit(normal.unitVector());
      float numerator = normal_unit.dot(a - ray_start);
      float denominator = normal_unit.dot(ray_unit);

      if (
         (fabs(denominator) <= 1e-7f) ||
         (std::signbit(numerator) != std::signbit(denominator))
      )
      {
         return false;
      }

      intersection = ray_start + ray_unit * (numerator / denominator);

      return true;
   }

   Vector3 projectPointToPlane(
      const geometry::types::plane_t & source,
      const geometry::types::plane_t & dest
   )
   {
      float t = (
         (dest.point - source.point).dot(dest.normal) / (source.normal.dot(dest.normal))
      );

      Vector3 projected_source_point = source.point + t * source.normal;

      return projected_source_point;
   }

   geometry::types::clippedSegment_t clipSegment(
      const Vector3 & a,
      const Vector3 & b,
      const geometry::types::plane_t & plane
   )
   {
      const float signed_distance_a = plane.normal.dot(a - plane.point);
      const float signed_distance_b = plane.normal.dot(b - plane.point);

      geometry::types::clippedSegment_t result;
      result.clipped = (
         (signed_distance_a >= 0.f) || (signed_distance_b >= 0.f)
      );

      if (!result.clipped)
      {
         return result;
      }

      if (std::signbit(signed_distance_a) == std::signbit(signed_distance_b))
      {
         result.points[0] = a;
         result.points[1] = b;
         return result;
      }

      const float t = (
         plane.normal.dot(plane.point - a) / plane.normal.dot(b - a)
      );

      result.points[0] = (signed_distance_a >= 0.f) ? a : b;
      result.points[1] = a + (b - a) * t;

      return result;
   }
}

}
