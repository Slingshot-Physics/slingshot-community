#ifndef PLANE_HEADER
#define PLANE_HEADER

#include "geometry_types.hpp"
#include "vector3.hpp"

namespace geometry
{

namespace plane
{

   // Find the point on a plane defined by a normal vector n and point a
   // closest to another point q.
   Vector3 closestPointToPoint(
      const Vector3 & n, const Vector3 & a, const Vector3 & q
   );

   // Find the point on a plane defined by points [a, b, c] closest to another
   // point q.
   Vector3 closestPointToPoint(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q
   );

   // Determines if a point q is coplaner with the plane defined by [a, b, c]
   // given some epsilon whose default is 2e-7f.
   bool pointCoplanar(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q,
      float epsilon=2e-7f
   );

   // Determine if a ray intersects with a plane.
   // The plane is defined by a normal, normal, and point, a.
   // The ray is defined by a start position, ray_start, and direction,
   // ray_unit.
   // The contact point is returned as an output arg.
   // Returns true if an intersection occurs, false otherwise. Only
   // intersections yielding single points are treated as true intersections.
   bool rayIntersect(
      const Vector3 & normal,
      const Vector3 & a,
      const Vector3 & ray_start,
      const Vector3 & ray_unit,
      Vector3 & intersection
   );

   // Projects the point in the 'source' plane onto the 'dest' plane by
   // offsetting the source point in the direction of the source plane's
   // normal. In short, the point on 'source' gets projected onto 'dest'.
   // Returns the projected point.
   Vector3 projectPointToPlane(
      const geometry::types::plane_t & source,
      const geometry::types::plane_t & dest
   );

   // Clips a segment against a plane. Only the clipped points that are in the
   // plane's positive half-space are kept.
   //
   // clippedSegment.clipped is true if the clipped segment's endpoints are in
   // the plane's positive half-space, false otherwise. E.g. if the end points
   // of the original segment are in the plane's negative half-space, then
   // clippedSegment.clipped will be false.
   //
   // clippedSegment.points contains the clipped endpoints of the segment that
   // are in the plane's positive half-space.
   //
   // No clipping occurs if the segment is below the plane.
   geometry::types::clippedSegment_t clipSegment(
      const Vector3 & a,
      const Vector3 & b,
      const geometry::types::plane_t & plane
   );

}

}

#endif
