#include "polygon.hpp"

#include "attitudeutils.hpp"
#include "line.hpp"
#include "mesh_ops.hpp"
#include "segment.hpp"
#include "transform.hpp"

#include <cmath>

namespace geometry
{

namespace polygon
{

   typedef geometry::types::polygon50_t polygon50_t;

   bool halfPlane(
      const Vector3 & a, const Vector3 & b, const Vector3 & q, bool eq
   )
   {
      if (eq)
      {
         return ((b - a).crossProduct(q - a))[2] >= 0.0f;
      }

      return ((b - a).crossProduct(q - a))[2] > 0.0f;
   }

   bool halfPlane(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & q,
      const Vector3 & n,
      bool eq
   )
   {
      if (eq)
      {
         return ((b - a).crossProduct(q - a)).dot(n) >= 0.0f;
      }

      return ((b - a).crossProduct(q - a)).dot(n) > 0.0f;
   }

   void convexIntersection(
      const polygon50_t & poly_p,
      const polygon50_t & poly_q,
      polygon50_t & poly_intersection
   )
   {
      poly_intersection.numVerts = 0;

      if (poly_p.numVerts >= 3 && poly_q.numVerts >= 3)
      {
         polygon50_t poly_p_ccw = poly_p;
         polygon50_t poly_q_ccw = poly_q;
         orderVertsCcw(poly_p_ccw);
         orderVertsCcw(poly_q_ccw);
         clipSutherlandHodgman(poly_p_ccw, poly_q_ccw, poly_intersection);
      }
      else if (poly_p.numVerts >= 3 && poly_q.numVerts == 2)
      {
         geometry::types::segment_t segment_q;
         segment_q.points[0] = poly_q.verts[0];
         segment_q.points[1] = poly_q.verts[1];

         geometry::types::segment_t clipped_segment;

         bool clipped = clipSegment(poly_p, segment_q, clipped_segment);

         if (clipped)
         {
            poly_intersection.numVerts = 2;
            poly_intersection.verts[0] = clipped_segment.points[0];
            poly_intersection.verts[1] = clipped_segment.points[1];
         }
      }
      else if (poly_p.numVerts == 2 && poly_q.numVerts >= 3)
      {
         geometry::types::segment_t segment_p;
         segment_p.points[0] = poly_p.verts[0];
         segment_p.points[1] = poly_p.verts[1];

         geometry::types::segment_t clipped_segment;

         bool clipped = clipSegment(poly_q, segment_p, clipped_segment);

         if (clipped)
         {
            poly_intersection.numVerts = 2;
            poly_intersection.verts[0] = clipped_segment.points[0];
            poly_intersection.verts[1] = clipped_segment.points[1];
         }
      }
      else if (poly_p.numVerts == 2 && poly_q.numVerts == 2)
      {
         Vector3 a, b;
         int num_intersections = geometry::segment::intersection(
            poly_p.verts[0],
            poly_p.verts[1],
            poly_q.verts[0],
            poly_q.verts[1],
            a,
            b
         );

         if (num_intersections > 0)
         {
            poly_intersection.numVerts = num_intersections;
            poly_intersection.verts[0] = a;
            poly_intersection.verts[1] = b;
         }
      }
      // Could continue to 2-1, 1-2, and 1-1, but I don't need to right now.
   }

   void clipSutherlandHodgman(
      const polygon50_t & poly_clip,
      const polygon50_t & poly_subject,
      polygon50_t & poly_clipped_subject
   )
   {
      if (poly_clip.numVerts < 3 || poly_subject.numVerts < 3)
      {
         poly_clipped_subject.numVerts = 0;
         return;
      }

      Vector3 center_poly_clip;
      for (unsigned int i = 0; i < poly_clip.numVerts; ++i)
      {
         center_poly_clip += poly_clip.verts[i] / (float )poly_clip.numVerts;
      }

      // The vector normal to the plane defined by the clip polygon. This is ok
      // because the function assumes that the vertices of the clip polygon are
      // oriented CW or CCW.
      Vector3 normal_poly_clip = (
         (poly_clip.verts[0] - center_poly_clip).crossProduct(
            poly_clip.verts[1] - center_poly_clip
         )
      );

      polygon50_t poly_output = poly_subject;
      unsigned int & numOutputVerts = poly_output.numVerts;

      for (unsigned int i = 0; i < poly_clip.numVerts; ++i)
      {
         polygon50_t poly_input = poly_output;
         numOutputVerts = 0;

         const Vector3 & clip_point1 = poly_clip.verts[(i - 1 + poly_clip.numVerts) % poly_clip.numVerts];
         const Vector3 & clip_point2 = poly_clip.verts[i];

         for (unsigned int j = 0; j < poly_input.numVerts; ++j)
         {
            const Vector3 & subj_point1 = poly_input.verts[(j - 1 + poly_input.numVerts) % poly_input.numVerts];
            const Vector3 & subj_point2 = poly_input.verts[j];

            Vector3 p, q;

            int intersectionType = geometry::line::segmentIntersection(
               clip_point1, clip_point2, subj_point1, subj_point2, p, q
            );

            if (intersectionType > 0)
            {
               poly_output.verts[numOutputVerts] = p;
               ++numOutputVerts;
               if (numOutputVerts == 49)
               {
                  return;
               }
            }

            if (
               halfPlane(
                  clip_point1, clip_point2, subj_point2, normal_poly_clip, true
               )
            )
            {
               poly_output.verts[numOutputVerts] = subj_point2;
               ++numOutputVerts;
               if (numOutputVerts == 49)
               {
                  return;
               }
            }
         }
      }

      // Delete any duplicated points.
      orderVertsCcw(poly_output);

      if (poly_output.numVerts == 0)
      {
         poly_clipped_subject = poly_output;
         return;
      }

      unsigned int & num_cs_verts = poly_clipped_subject.numVerts;
      num_cs_verts = 1;
      poly_clipped_subject.verts[0] = poly_output.verts[0];

      for (unsigned int i = 1; i < poly_output.numVerts; ++i)
      {
         Vector3 & temp_vert = poly_output.verts[i % poly_output.numVerts];
         if (
            (
               poly_clipped_subject.verts[num_cs_verts - 1] - temp_vert
            ).magnitude() > 1e-6f
         )
         {
            poly_clipped_subject.verts[num_cs_verts] = temp_vert;
            ++num_cs_verts;
         }
      }
   }

   bool clipSegment(
      const geometry::types::polygon50_t & poly_clip,
      const geometry::types::segment_t & segment_subject,
      geometry::types::segment_t & clipped_segment
   )
   {
      if (poly_clip.numVerts < 3)
      {
         return false;
      }

      if (
         pointInside(poly_clip, segment_subject.points[0]) && 
         pointInside(poly_clip, segment_subject.points[1])
      )
      {
         clipped_segment = segment_subject;
         return true;
      }

      geometry::types::polygon50_t poly_clip_ccw(poly_clip);
      orderVertsCcw(poly_clip_ccw);

      clipped_segment = segment_subject;

      bool segment_clipped = false;

      for (unsigned int i = 0; i < poly_clip_ccw.numVerts; ++i)
      {
         Vector3 p, q;
         int num_intersections = geometry::segment::intersection(
            poly_clip_ccw.verts[i],
            poly_clip_ccw.verts[(i + 1) % poly_clip_ccw.numVerts],
            clipped_segment.points[0],
            clipped_segment.points[1],
            p,
            q,
            1e-6f
         );

         segment_clipped |= (num_intersections > 0);

         switch (num_intersections)
         {
            case 0:
               break;
            case 1:
            {
               bool seg_points_inside[2] = {
                  halfPlane(
                     poly_clip_ccw.verts[i],
                     poly_clip_ccw.verts[(i + 1) % poly_clip_ccw.numVerts],
                     clipped_segment.points[0],
                     true
                  ),
                  halfPlane(
                     poly_clip_ccw.verts[i],
                     poly_clip_ccw.verts[(i + 1) % poly_clip_ccw.numVerts],
                     clipped_segment.points[1],
                     true
                  )
               };

               // segment_outside_polygon = false;
               if (!seg_points_inside[0])
               {
                  clipped_segment.points[0] = p;
               }
               else if (!seg_points_inside[1])
               {
                  clipped_segment.points[1] = p;
               }
               break;
            }
            case 2:
            {
               // segment_outside_polygon = false;
               clipped_segment.points[0] = p;
               clipped_segment.points[1] = q;
               break;
            }
            default:
               break;
         }
      }

      return segment_clipped;
   }

   void convexHull(
      const polygon50_t & poly_in,
      polygon50_t & poly_convex
   )
   {
      // Find the vertex that's in the "lower-left-most" corner.
      Vector3 start_point(poly_in.verts[0]);
      for (unsigned int i = 0; i < poly_in.numVerts; ++i)
      {
         if (poly_in.verts[i][1] < start_point[1])
         {
            start_point = poly_in.verts[i];
         }
         else if (poly_in.verts[i][1] == start_point[1])
         {
            if (poly_in.verts[i][0] < start_point[0])
            {
               start_point = poly_in.verts[i];
            }
         }
      }

      polygon50_t temp_poly = poly_in;

      // Order the vertices relative to the angle they make with the start
      // vertex.
      for (unsigned int i = 0; i < temp_poly.numVerts; ++i)
      {
         Vector3 & temp_vert = temp_poly.verts[i];
         Vector3 unit_vec;
         float temp_vert_dist = (temp_vert - start_point).magnitude();
         if (temp_vert_dist < 1e-7f)
         {
            unit_vec[0] = -1.f / sqrtf(2.f);
            unit_vec[1] = -1.f / sqrtf(2.f);
            unit_vec[2] = 0.f;
         }
         else
         {
            float denom = std::fmax(temp_vert_dist, 1e-7f);
            unit_vec = (temp_vert - start_point) / denom;
         }

         temp_vert[2] = atan2(unit_vec[1], unit_vec[0]);
      }

      std::sort(
         temp_poly.verts,
         temp_poly.verts + temp_poly.numVerts,
         polyComparator_t()
      );

      // The Graham-scan-ification. Iterate over sequential points on the
      // polygon, only keep sequences of points that "turn left".
      unsigned int & num_convex_verts = poly_convex.numVerts;
      num_convex_verts = 0;
      for (unsigned int i = 0; i < temp_poly.numVerts; ++i)
      {
         Vector3 & point = temp_poly.verts[i];
         if (num_convex_verts < 2)
         {
            poly_convex.verts[num_convex_verts] = point;
            ++num_convex_verts;

            if (num_convex_verts == 49)
            {
               return;
            }
         }
         else
         {
            bool last_popped = false;
            do
            {
               last_popped = false;
               Vector3 prev_pt = poly_convex.verts[num_convex_verts - 1];
               Vector3 penult_pt = poly_convex.verts[num_convex_verts - 2];

               if ((point - prev_pt).crossProduct(prev_pt - penult_pt)[2] >= 0.f)
               {
                  last_popped = true;
                  num_convex_verts -= 1;
               }
            } while (last_popped && num_convex_verts > 2);

            poly_convex.verts[num_convex_verts] = point;
            ++num_convex_verts;

            if (num_convex_verts == 49)
            {
               return;
            }
         }
      }
   }

   bool pointInside(
      const geometry::types::polygon50_t & poly, const Vector3 & q
   )
   {
      geometry::types::polygon50_t poly_ccw = poly;
      orderVertsCcw(poly_ccw);

      for (unsigned int i = 0; i < poly_ccw.numVerts; ++i)
      {
         if (!halfPlane(poly_ccw.verts[i], poly_ccw.verts[(i + 1) % poly_ccw.numVerts], q))
         {
            return false;
         }
      }

      return true;
   }

   // Assumes that the z-index of each vertex is zero, which allows the
   // z-index to be populated by sortable values.
   void orderVertsCcw(
      const Vector3 * unordered_verts,
      unsigned int num_verts,
      Vector3 * ordered_verts
   )
   {
      Vector3 center = averageVertex(num_verts, unordered_verts);
      center[2] = 0.f;

      for (unsigned int i = 0; i < num_verts; ++i)
      {
         ordered_verts[i] = unordered_verts[i];
      }

      // Get order-preserving estimate of 1/pi of the angle between the
      // (vertex - center) and the x-axis. Shove the estimate in the z-value.
      for (unsigned int i = 0; i < num_verts; ++i)
      {
         float x = ordered_verts[i][0] - center[0];
         float y = ordered_verts[i][1] - center[1];
         ordered_verts[i][2] = atan2(y, x);
      }

      std::sort(
         ordered_verts, ordered_verts + num_verts, polyComparator_t()
      );

      // Set the z-values of the ordered polygon vertices back to zero.
      for (unsigned int i = 0; i < num_verts; ++i)
      {
         ordered_verts[i][2] = 0.0f;
      }
   }

   void orderVertsCcw(polygon50_t & convexPoly)
   {
      Vector3 center;
      averageVertex(convexPoly.numVerts, convexPoly.verts, center);

      polygon50_t tempPoly;
      tempPoly.numVerts = convexPoly.numVerts;
      // Get order-preserving estimate of 1/pi of the angle between the
      // (vertex - center) and the x-axis. Shove the estimate in the z-value.
      for (unsigned int i = 0; i < convexPoly.numVerts; ++i)
      {
         float x = convexPoly.verts[i][0] - center[0];
         float y = convexPoly.verts[i][1] - center[1];
         tempPoly.verts[i] = convexPoly.verts[i];
         tempPoly.verts[i][2] = atan2(y, x);
      }

      std::sort(
         tempPoly.verts,
         tempPoly.verts + tempPoly.numVerts,
         polyComparator_t()
      );

      // Set the z-values of the ordered polygon vertices back to zero.
      for (unsigned int i = 0; i < convexPoly.numVerts; ++i)
      {
         convexPoly.verts[i] = tempPoly.verts[i];
         convexPoly.verts[i][2] = 0.0f;
      }
   }

   void applyTransformation(
      const geometry::types::isometricTransform_t & transform,
      geometry::types::polygon50_t & polygon_in
   )
   {
      for (unsigned int i = 0; i < polygon_in.numVerts; ++i)
      {
         polygon_in.verts[i] = geometry::transform::forwardBound(
            transform, polygon_in.verts[i]
         );
      }
   }

   void applyTransformation(
      const geometry::types::transform_t & transform,
      geometry::types::polygon50_t & polygon_in
   )
   {
      for (unsigned int i = 0; i < polygon_in.numVerts; ++i)
      {
         polygon_in.verts[i] = geometry::transform::forwardBound(
            transform, polygon_in.verts[i]
         );
      }
   }

   geometry::types::polygon50_t projectPolygonToPlane(
      const geometry::types::polygon50_t & polygon,
      const Vector3 & normal,
      const geometry::types::plane_t & dest
   )
   {
      geometry::types::polygon50_t projected_polygon;
      projected_polygon.numVerts = polygon.numVerts;

      float numerator_a = dest.point.dot(dest.normal);
      float denom = normal.dot(dest.normal);

      for (unsigned int i = 0; i < polygon.numVerts; ++i)
      {
         float numerator_b = polygon.verts[i].dot(dest.normal);
         float t = (numerator_a - numerator_b) / denom;
         projected_polygon.verts[i] = polygon.verts[i] + t * normal;
      }

      return projected_polygon;
   }

   geometry::types::polygon4_t projectPolygonToPlane(
      const geometry::types::polygon4_t & polygon,
      const Vector3 & normal,
      const geometry::types::plane_t & dest
   )
   {
      geometry::types::polygon4_t projected_polygon;
      projected_polygon.numVerts = polygon.numVerts;

      float numerator_a = dest.point.dot(dest.normal);
      float denom = normal.dot(dest.normal);

      for (unsigned int i = 0; i < polygon.numVerts; ++i)
      {
         float numerator_b = polygon.verts[i].dot(dest.normal);
         float t = (numerator_a - numerator_b) / denom;
         projected_polygon.verts[i] = polygon.verts[i] + t * normal;
      }

      return projected_polygon;
   }

   Vector3 calculateNormal(const geometry::types::polygon50_t & polygon)
   {
      Vector3 unit_normal;
      if (polygon.numVerts < 3)
      {
         return unit_normal;
      }

      Vector3 center = averageVertex(polygon.numVerts, polygon.verts);

      unit_normal = (polygon.verts[0] - center).crossProduct(
         polygon.verts[2] - center
      );

      unit_normal.unitVector();

      return unit_normal;
   }

}

}
