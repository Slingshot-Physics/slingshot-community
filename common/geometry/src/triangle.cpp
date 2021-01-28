#include "triangle.hpp"

#include "plane.hpp"
#include "segment.hpp"

#include <cmath>

namespace geometry
{

namespace triangle
{
   Vector4 baryCoords(const triangle_t & tri, const Vector3 & q)
   {
      return baryCoords(tri.verts[0], tri.verts[1], tri.verts[2], q);
   }

   Vector4 baryCoords(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q
   )
   {
      const Vector3 v0 = b - a;
      const Vector3 v1 = c - a;
      const Vector3 v2 = q - a;

      const Vector3 v0_unit = v0.unitVector();
      const Vector3 v1_unit = v1.unitVector();

      const float v0m = v0.magnitude();
      const float v1m = v1.magnitude();
      const float v0dv1u = v0.dot(v1_unit);
      const float v1dv0u = v1.dot(v0_unit);
      const float v2dv0u = v2.dot(v0_unit);
      const float v2dv1u = v2.dot(v1_unit);

      const float det = v0m * v1m - v1dv0u * v0dv1u;

      Vector4 bary;
      const float lambda_0 = v1m * v2dv0u - v2dv1u * v1dv0u;
      const float lambda_1 = v0m * v2dv1u - v0dv1u * v2dv0u;

      bary[1] = lambda_0 / det;
      bary[2] = lambda_1 / det;
      bary[0] = 1.f - bary[1] - bary[2];

      return bary;
   }

   pointBaryCoord_t closestPointToPoint(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q
   )
   {
      pointBaryCoord_t closest_point;

      if (
         ((a - b).magnitude() < 1e-6f) &&
         ((c - b).magnitude() < 1e-6f)
      )
      {
         // All vertices are coincident.
         closest_point.bary.Initialize(1.f, 0.f, 0.f, 0.f);
         closest_point.point = a;
         return closest_point;
      }
      else if (
         ((a - b).magnitude() < 1e-6f) ||
         ((c - b).magnitude() < 1e-6f)
      )
      {
         // Vertices a and b are coincident or vertices c and b are coincident
         // so the segment [a, c] defines the degenerate triangle.
         float t = (q - a).dot(c - a) / (c - a).magnitudeSquared();
         t = std::fmax(
            std::fmin(t, 1.f),
            0.f
         );
         closest_point.bary.Initialize(1.f - t, 0.f, t, 0.f);
         closest_point.point = (1.f - t) * a + t * c;
         return closest_point;
      }
      else if ((c - a).magnitude() < 1e-6f)
      {
         // Vertices a and c are coincident so the segment [a, b] defines the
         // degenerate triangle.
         float t = (q - a).dot(b - a) / (b - a).magnitudeSquared();
         t = std::max(
            std::min(t, 1.f),
            0.f
         );
         closest_point.bary.Initialize(1.f - t, t, 0.f, 0.f);
         closest_point.point = (1.f - t) * a + t * b;
         return closest_point;
      }

      Vector3 u(q - a);
      Vector3 x(b - a);
      Vector3 y(c - a);
      Vector3 z(b - c);

      // Vertices are colinear without any vertices being coincident
      if (x.unitVector().dot(y.unitVector()) >= (1.f - 1e-5f))
      {
         const float ydx = y.dot(x);
         const float xm2 = x.magnitudeSquared();
         const float ym2 = y.magnitudeSquared();
         const float zm2 = z.magnitudeSquared();
         if (ydx >= 0.f && ydx <= xm2)
         {
            // closest point from q to segment [a, b]
            const float t = std::max(
               std::min(
                  (q - a).dot(b - a) / xm2,
                  1.f
               ),
               0.f
            );

            closest_point.bary.Initialize(1.f - t, t, 0.f, 0.f);
            closest_point.point = (1.f - t) * a + t * b;
            return closest_point;
         }
         else if (ydx >= 0.f && ydx <= ym2)
         {
            // closest point from q to segment [a, c]
            const float t = std::max(
               std::min(
                  (q - a).dot(c - a) / ym2,
                  1.f
               ),
               0.f
            );

            closest_point.bary.Initialize(1.f - t, 0.f, t, 0.f);
            closest_point.point = (1.f - t) * a + t * c;
            return closest_point;
         }

         // closest point from q to segment [b, c]
         const float t = std::max(
            std::min(
               (q - b).dot(c - b) / zm2,
               1.f
            ),
            0.f
         );

         closest_point.bary.Initialize(0.f, 1.f - t, t, 0.f);
         closest_point.point = (1.f - t) * b + t * c;
         return closest_point;
      }

      float udx = u.dot(x);
      float udy = u.dot(y);

      // Vertex A is the VFR
      if (udx <= 0.f && udy <= 0.f)
      {
         closest_point.bary.Initialize(1.f, 0.f, 0.f, 0.f);
         closest_point.point = a;
         return closest_point;
      }

      Vector3 v(q - b);

      float vdx = v.dot(x);
      float vdz = v.dot(z);

      // Vertex B is the VFR
      if (vdx >= 0.f && vdz >= 0.f)
      {
         closest_point.bary.Initialize(0.f, 1.f, 0.f, 0.f);
         closest_point.point = b;
         return closest_point;
      }

      Vector3 w(q - c);

      float wdy = w.dot(y);
      float wdz = w.dot(z);

      // Vertex C is the VFR
      if (wdy >= 0.f && wdz <= 0.f)
      {
         closest_point.bary.Initialize(0.f, 0.f, 1.f, 0.f);
         closest_point.point = c;
         return closest_point;
      }

      Vector3 n(x.crossProduct(y));

      float vdy = v.dot(y);

      // Segment [A, B] is the VFR
      if ((udx >= 0.f) && (vdx <= 0.f) && ((vdx * udy - vdy * udx) >= 0.f))
      {
         float t_ab = std::fmin(std::fmax(udx / x.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(1.f - t_ab, t_ab, 0.f, 0.f);
         closest_point.point = (1.f - t_ab) * a + t_ab * b;
         return closest_point;
      }

      float wdx = w.dot(x);

      // Segment [B, C] is the VFR
      if ((vdz <= 0.f) && (wdz >= 0.f) && ((wdx * vdy - wdy * vdx) >= 0.f))
      {
         float t_bc = std::fmin(std::fmax(-1.f * vdz / z.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(0.f, 1.f - t_bc, t_bc, 0.f);
         closest_point.point = (1.f - t_bc) * b + t_bc * c;
         return closest_point;
      }

      // Segment [A, C] is the VFR
      if ((wdy <= 0.f) && (udy >= 0.f) && ((udx * wdy - udy * wdx) >= 0.f))
      {
         float t_ac = std::fmin(std::fmax(udy / y.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(1.f - t_ac, 0.f, t_ac, 0.f);
         closest_point.point = (1.f - t_ac) * a + t_ac * c;
         return closest_point;
      }

      // The entire triangle is the VFR
      closest_point.point = q - n * (n.dot(u) / n.magnitudeSquared());
      closest_point.bary = baryCoords(a, b, c, closest_point.point);

      // Fail-over recalculation of the closest point and bary
      // coordinates. Empirically, this code is called rarely and only in
      // cases where closest point queries are made against long, skinny
      // triangles.
      Vector4 & bary = closest_point.bary;
      if (
         (bary[0] < 0.f || bary[0] > 1.f) ||
         (bary[1] < 0.f || bary[1] > 1.f) ||
         (bary[2] < 0.f || bary[2] > 1.f)
      )
      {
         int voronoi_verts = 0;
         // Vertex A is on the Voronoi region
         voronoi_verts |= (bary[0] >= 0.f) << 0;
         // Vertex B is on the Voronoi region
         voronoi_verts |= (bary[1] >= 0.f) << 1;
         // Vertex C is on the Voronoi region
         voronoi_verts |= (bary[2] >= 0.f) << 2;

         // We know that only two of these vertices will be on the
         // Voronoi region, otherwise we wouldn't have entered this
         // if statement.
         switch(voronoi_verts)
         {
            // Vertex A is the Voronoi feature region.
            case 1:
            {
               closest_point.point = a;
               closest_point.bary.Initialize(1.f, 0.f, 0.f, 0.f);
               break;
            }
            // Vertex B is the Voronoi feature region.
            case 2:
            {
               closest_point.point = b;
               closest_point.bary.Initialize(0.f, 1.f, 0.f, 0.f);
               break;
            }
            // Segment [A, B] is the Voronoi feature region.
            case 3:
            {
               float t_ab = udx / x.magnitudeSquared();
               closest_point.bary.Initialize(1.f - t_ab, t_ab, 0.f, 0.f);
               closest_point.point = (1.f - t_ab) * a + t_ab * b;
               break;
            }
            // Vertex C is the Voronoi feature region.
            case 4:
            {
               closest_point.point = c;
               closest_point.bary.Initialize(0.f, 0.f, 1.f, 0.f);
               break;
            }
            // Segment [A, C] is the Voronoi feature region.
            case 5:
            {
               float t_ac = udy / y.magnitudeSquared();
               closest_point.bary.Initialize(1.f - t_ac, 0.f, t_ac, 0.f);
               closest_point.point = (1.f - t_ac) * a + t_ac * c;
               break;
            }
            // Segment [B, C] is the Voronoi feature region.
            case 6:
            {
               float t_bc = -1.f * vdz / z.magnitudeSquared();
               closest_point.bary.Initialize(0.f, 1.f - t_bc, t_bc, 0.f);
               closest_point.point = (1.f - t_bc) * b + t_bc * c;
               break;
            }
            default:
               break;
         }
      }

      return closest_point;
   }

   bool pointInFaceVfr(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q
   )
   {
      Vector3 normal = (a - b).crossProduct(c - b);

      Vector3 plane_ab_normal = normal.crossProduct(a - b);
      Vector3 plane_bc_normal = normal.crossProduct(b - c);
      Vector3 plane_ca_normal = normal.crossProduct(c - a);

      // Make all the plane normals point toward the center of the triangle
      plane_ab_normal *= (plane_ab_normal.dot(c - a) > 0.f) ? 1.f : -1.f;
      plane_bc_normal *= (plane_bc_normal.dot(a - b) > 0.f) ? 1.f : -1.f;
      plane_ca_normal *= (plane_ca_normal.dot(b - c) > 0.f) ? 1.f : -1.f;

      return (
         (plane_ab_normal.dot(q - a) >= 0.f) &&
         (plane_bc_normal.dot(q - b) >= 0.f) &&
         (plane_ca_normal.dot(q - c) >= 0.f)
      );
   }

   bool pointCoplanar(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & q,
      float epsilon
   )
   {
      Vector3 p = closestPointToPoint(a, b, c, q).point;

      return (q - p).magnitudeSquared() <= epsilon;
   }

   bool rayIntersect(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & ray_start,
      const Vector3 & ray_unit,
      Vector3 & intersection
   )
   {
      Vector3 triangle_unit(((a - b).crossProduct(c - b)).unitVector());
      if (
         geometry::plane::rayIntersect(
            triangle_unit, a, ray_start, ray_unit, intersection
         )
      )
      {
         geometry::types::pointBaryCoord_t closest_bary = closestPointToPoint(
            a, b, c, intersection
         );

         if ((closest_bary.point - intersection).magnitude() < 1e-6f)
         {
            return true;
         }
         else
         {
            return false;
         }
      }

      return false;
   }

   float area(const Vector3 & a, const Vector3 & b, const Vector3 & c)
   {
      float area = 0.5f * ((c - a).crossProduct(b - a)).magnitude();

      return area;
   }

}

}
