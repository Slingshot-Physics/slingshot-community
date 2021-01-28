#ifndef TRIANGLE_HEADER
#define TRIANGLE_HEADER

#include "geometry_types.hpp"

namespace geometry
{

namespace triangle
{

   typedef geometry::types::pointBaryCoord_t pointBaryCoord_t;
   typedef geometry::types::triangle_t triangle_t;

   // Return the barycentric coordinates of a point, q, on the triangle, tri.
   Vector4 baryCoords(const triangle_t & tri, const Vector3 & q);

   // Return the barycentric coordinates of a point, q, on the triangle
   // [a, b, c].
   Vector4 baryCoords(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q
   );

   // Returns the point on triangle [a, b, c] closest to point q.
   pointBaryCoord_t closestPointToPoint(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q
   );

   // Returns true if a point q is in the face Voronoi feature region of the
   // triangle defined by [a, b, c].
   bool pointInFaceVfr(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q
   );

   // Returns true if a point q is coplanar with the triangle defined by
   // [a, b, c].
   bool pointCoplanar(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q,
      float epsilon=1e-6f
   );

   // Determines if a ray defined by [ray_start, ray_unit] intersects the
   // triangle defined by [a, b, c]. Returns true if an intersection occurs,
   // false otherwise. Only intersections yielding single points are treated
   // as true intersections.
   bool rayIntersect(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & ray_start,
      const Vector3 & ray_unit,
      Vector3 & intersection
   );

   float area(const Vector3 & a, const Vector3 & b, const Vector3 & c);

}

}

#endif
