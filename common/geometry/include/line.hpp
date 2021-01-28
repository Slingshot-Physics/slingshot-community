#ifndef LINE_HEADER
#define LINE_HEADER

#include "vector3.hpp"

namespace geometry
{

namespace line
{

   // Finds the parameter from r = t * a + (1 - t) * b that generates the point
   // on the line passing through points [a, b] closest to point c.
   float closestParameterToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & q
   );

   // Find the point on the line passing through a and b closest to point c.
   Vector3 closestPointToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & q
   );

   // Calculates the closest points between two lines, the lines passing
   // through [a, b] and [c, d]. Point p is the point on [a, b] closest to
   // [c, d], and point q is the point on [c, d] closest to [a, b].
   // It's guaranteed that there is at least one closest point between any two
   // lines.
   // Returns 1 if there is one pair of closest points between the two lines.
   // Returns 2 if there is more than one pair of closest points between the
   // two lines.
   // If there is more than one pair of closest points, p and q are set to a
   // and c, respectively.
   int closestPointsToLine(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      Vector3 & p,
      Vector3 & q,
      float epsilon=1e-6f
   );

   // Tests if point c is colinear with the line passing through [a, b].
   bool pointColinear(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & q,
      float epsilon=1e-6f
   );

   // Calculates the type and points of intersection between a line and a line
   // segment. The line is defined as passing through the points [a, b] and the
   // segment is defined as passing through [c, d].
   // Returns 0 and does not set p or q if there are no intersections.
   // Returns 1 and sets p = intersection point if there is one intersection.
   // Returns 2 and sets p = min intersection, q = max intersection if there
   // are infinite intersections.
   int segmentIntersection(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      Vector3 & p,
      Vector3 & q
   );

} // line

} // geometry

#endif
