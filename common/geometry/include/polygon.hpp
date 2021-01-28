#ifndef POLYGON_HEADER
#define POLYGON_HEADER

#include "geometry_types.hpp"
#include "vector3.hpp"

namespace geometry
{

namespace polygon
{
   struct polyComparator_t
   {
      bool operator()(const Vector3 & a, const Vector3 & b) const
      {
         return a[2] < b[2];
      }
   };

   // For two points on a 2D segment in the XY plane, determine if a third
   // point in the XY plane is in the positive half space of segment [a, b].
   bool halfPlane(
      const Vector3 & a, const Vector3 & b, const Vector3 & q, bool eq=true
   );

   // For two points on an infinite line in 3D space, [a, b], determine if a
   // query point, q, is in the positive half space defined by the normal n and
   // the line [a, b].
   bool halfPlane(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & q,
      const Vector3 & n,
      bool eq=true
   );

   // Makes one call to the Sutherland-Hodgman algorithm to find the polygon
   // of intersection between two convex polygons, P and Q, that are both on
   // the x-y plane (the z component of the polygons must be zero).

   // Find the intersection between two polygons with two or more vertices that
   // are both on the x-y plane.
   void convexIntersection(
      const geometry::types::polygon50_t & poly_p,
      const geometry::types::polygon50_t & poly_q,
      geometry::types::polygon50_t & poly_intersection
   );

   // Clips a subject polygon using the edges of a convex "clipping" polygon.
   // Assumes that the vertices in the polygons are ordered either clockwise or
   // CCW (doesn't matter which one as long as they're both the same).
   // Implementation based on Sutherland-Hodgman clipping algorithm.
   // Both polygons must be on the x-y plane (e.g. their z-components must be
   // zero).
   void clipSutherlandHodgman(
      const geometry::types::polygon50_t & poly_clip,
      const geometry::types::polygon50_t & poly_subject,
      geometry::types::polygon50_t & poly_clipped_subject
   );

   // Returns true if the segment is clipped by the polygon, false otherwise.
   bool clipSegment(
      const geometry::types::polygon50_t & poly_clip,
      const geometry::types::segment_t & segment_subject,
      geometry::types::segment_t & clipped_segment
   );

   // Uses Graham scans to generate a convex polygon out of an arbitrary
   // polygon. Accepts a struct containing arrays of Vector3's, but assumes
   // that only the x and y components of the vertices contribute to the
   // polygon's structure.
   void convexHull(
      const geometry::types::polygon50_t & poly_in,
      geometry::types::polygon50_t & poly_convex
   );

   // Determines if the point q is inside a convex polygon. Returns true if q
   // is inside 'poly', false otherwise.
   bool pointInside(
      const geometry::types::polygon50_t & poly, const Vector3 & q
   );

   // Order the vertices of a convex polygon with vertices in the XY plane in a
   // CCW rotation.
   void orderVertsCcw(
      const Vector3 * unordered_verts,
      unsigned int num_verts,
      Vector3 * ordered_verts
   );

   // Order the vertices of a convex polygon with vertices in the XY plane in a
   // CCW rotation.
   void orderVertsCcw(geometry::types::polygon50_t & polygon);

   // The input polygon is the output polygon. Its vertices are transformed
   // according to the transformation in the function argument.
   void applyTransformation(
      const geometry::types::isometricTransform_t & transform,
      geometry::types::polygon50_t & polygon_in
   );

   // The input polygon is the output polygon. Its vertices are transformed
   // according to the transformation in the function argument.
   void applyTransformation(
      const geometry::types::transform_t & transform,
      geometry::types::polygon50_t & polygon_in
   );

   // Projects the points on the 'polygon' onto the 'dest' plane by offsetting
   // the points on the polygon in the direction of 'normal'.
   geometry::types::polygon50_t projectPolygonToPlane(
      const geometry::types::polygon50_t & polygon,
      const Vector3 & normal,
      const geometry::types::plane_t & dest
   );

   // Projects the points on the 'polygon' onto the 'dest' plane by offsetting
   // the points on the polygon in the direction of 'normal'.
   geometry::types::polygon4_t projectPolygonToPlane(
      const geometry::types::polygon4_t & polygon,
      const Vector3 & normal,
      const geometry::types::plane_t & dest
   );

   // Calculates the unit normal of the polygon if the polygon contains more
   // than three vertices. A zero vector is returned if the polygon has fewer
   // than three vertices. Assumes that the vertices of the polygon are not
   // colinear and that none of the vertices are degenerate.
   Vector3 calculateNormal(const geometry::types::polygon50_t & polygon);

}

}

#endif
