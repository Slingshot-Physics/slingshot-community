#ifndef SEGMENT_HEADER
#define SEGMENT_HEADER

#include "geometry_types.hpp"

namespace geometry
{

namespace segment
{
   typedef geometry::types::pointBaryCoord_t pointBaryCoord_t;

   // Finds the parameter from r = t * a + (1 - t) * b that generates the point on
   // the segment [a, b] closest to point q.
   float closestParameterToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & q
   );

   // Find the point on the segment between a and b closest to point q.
   pointBaryCoord_t closestPointToPoint(
      const Vector3 & a, const Vector3 & b, const Vector3 & q
   );

   // Find the points on the segment [a, b] closest to the segment [c, d].
   // Returns 1 or 2 pairs of closest points.
   // numPoints = 1 if there is 1 pair of closest points. segmentPoints[0] is
   // set to the closest point on [a, b] to [c, d], and otherPoints[0] is set
   // to the closest point on [c, d] to [a, b].
   // numPoints = 2 if the two segments are parallel and overlapping. If the
   // segments are overlapping and parallel, then there is a continuum of
   // closest points. segmentPoints[0] and segmentPoints[1] mark the start and
   // end points of the overlap from segment [a, b]. otherPoints[0] and
   // otherPoints[1] mark the start and end points of the overlap from segment
   // [c, d].
   // epsilon is a margin for parallelism tolerance of the segments. epsilon
   // is clamped between [0, 0.1].
   geometry::types::segmentClosestPoints_t closestPointsToSegment(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      float epsilon
   );

   // Finds the closest points between a segment [a, b] and a plane. If the
   // segment is parallel to the plane, then otherPoints[0] and otherPoints[1]
   // mark the start and end closest points between the plane and [a, b].
   // If the segment is not parallel to the plane, then only one pair of
   // closest points is returned.
   geometry::types::segmentClosestPoints_t closestPointsToPlane(
      const Vector3 & a,
      const Vector3 & b,
      const geometry::types::plane_t & plane
   );

   // Finds the closest points between a segment [a, b] and an AABB. If there
   // are multiple closest points, then the endpoints of the set of closest
   // points is returned.
   geometry::types::segmentClosestPoints_t closestPointsToAabb(
      const Vector3 & a,
      const Vector3 & b,
      const geometry::types::aabb_t & aabb
   );

   // Clips a segment [a, b] against the planes bounding an axis-aligned
   // square whose corners are on the AABB and whose face normal is a one-hot
   // vector with index 'face_normal_index' set to +/- 1.
   // Returns a clipped line segment type. If the segment [a, b] is inside the
   // bounded region, then the 'clipped' field will be set to true.
   geometry::types::clippedSegment_t clipSegmentAgainstAxisAlignedSquare(
      const Vector3 & a,
      const Vector3 & b,
      const geometry::types::aabb_t & aabb,
      const int face_normal_index
   );

   // Returns true if the point q is colinear with the segment [a, b], false
   // otherwise.
   bool pointColinear(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & q,
      float epsilon = 1e-7f
   );

   // Returns 1 if the two segments [a, b] and [c, d] intersect at one point, and
   // the intersection point is set in argument p.
   // Returns 2 if the two segments overlap. Puts the two endpoints of
   // intersection into arguments p and q.
   // Returns 0 if the two segments do not overlap.
   int intersection(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      Vector3 & p,
      Vector3 & q,
      float epsilon = 1e-14f
   );

   float length(const Vector3 & a, const Vector3 & b);

}

}

#endif
