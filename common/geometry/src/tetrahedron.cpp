#include "tetrahedron.hpp"

#include "matrix33.hpp"
#include "triangle.hpp"

#include <cmath>

namespace geometry
{

namespace tetrahedron
{
   Vector4 baryCoords(const tetrahedron_t & tetra, const Vector3 & q)
   {
      return baryCoords(
         tetra.verts[0], tetra.verts[1], tetra.verts[2], tetra.verts[3], q
      );
   }

   Vector4 baryCoords(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      const Vector3 & q
   )
   {
      Vector3 vda = a - d;
      Vector3 vdb = b - d;
      Vector3 vdc = c - d;
      Matrix33 T(
         vda[0], vdb[0], vdc[0],
         vda[1], vdb[1], vdc[1],
         vda[2], vdb[2], vdc[2]
      );
      Vector3 bary3 = (~T) * (q - d);

      Vector4 bary;

      for (int i = 0; i < 3; ++i)
      {
         bary[i] = bary3[i];
      }
      bary[3] = 1.f - bary[0] - bary[1] - bary[2];

      return bary;
   }

   pointBaryCoord_t closestPointToPoint(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      const Vector3 & q
   )
   {
      pointBaryCoord_t closest_point;

      Vector3 e(q - a);
      Vector3 f(q - b);
      Vector3 g(q - c);
      Vector3 h(q - d);

      Vector3 u(b - a);
      Vector3 v(d - a);
      Vector3 w(c - a);
      Vector3 x(c - b);
      Vector3 y(d - b);
      Vector3 z(d - c);

      float edu = e.dot(u);
      float edv = e.dot(v);
      float edw = e.dot(w);

      // Vertex A is the VFR
      if (edu <= 0.f && edv <= 0.f && edw <= 0.f)
      {
         closest_point.bary.Initialize(1.f, 0.f, 0.f, 0.f);
         closest_point.point = a;
         return closest_point;
      }

      float fdx = f.dot(x);
      float fdu = f.dot(u);
      float fdy = f.dot(y);

      // Vertex B is the VFR
      if (fdx <= 0.f && fdu >= 0.f && fdy <= 0.f)
      {
         closest_point.bary.Initialize(0.f, 1.f, 0.f, 0.f);
         closest_point.point = b;
         return closest_point;
      }

      float gdw = g.dot(w);
      float gdx = g.dot(x);
      float gdz = g.dot(z);

      // Vertex C is the VFR
      if (gdw >= 0.f && gdx >= 0.f && gdz <= 0.f)
      {
         closest_point.bary.Initialize(0.f, 0.f, 1.f, 0.f);
         closest_point.point = c;
         return closest_point;
      }

      float hdv = h.dot(v);
      float hdy = h.dot(y);
      float hdz = h.dot(z);

      // Vertex D is the VFR
      if (hdv >= 0.f && hdy >= 0.f && hdz >= 0.f)
      {
         closest_point.bary.Initialize(0.f, 0.f, 0.f, 1.f);
         closest_point.point = d;
         return closest_point;
      }

      Vector3 nDAB = v.crossProduct(u);
      Vector3 nBAC = u.crossProduct(x);
      Vector3 tDAB = nDAB.crossProduct(u);
      Vector3 tBAC = u.crossProduct(nBAC);

      // Edge [A, B] is the VFR
      if ((e.dot(tBAC) >= 0.f) && (e.dot(tDAB) >= 0.f) && (e.dot(u) >= 0.f) && (f.dot(u) <= 0.f))
      {
         float t_ab = std::fmin(std::fmax(e.dot(u)/u.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(1.f - t_ab, t_ab, 0.f, 0.f);
         closest_point.point = (1.f - t_ab) * a + t_ab * b;
         return closest_point;
      }

      Vector3 nBCD = -1.f * x.crossProduct(z);
      Vector3 tACBxbc = -1.f * nBAC.crossProduct(x);
      Vector3 tbcxBCD = -1.f * x.crossProduct(nBCD);

      // Edge [B, C] is the VFR
      if ((g.dot(tACBxbc) >= 0.f) && (g.dot(tbcxBCD) >= 0.f) && (f.dot(x) >= 0.f) && (g.dot(x) <= 0.f))
      {
         float t_bc = std::fmin(std::fmax(f.dot(x) / x.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(0.f, 1.f - t_bc, t_bc, 0.f);
         closest_point.point = (1.f - t_bc) * b + t_bc * c;
         return closest_point;
      }

      Vector3 nADC = v.crossProduct(z);
      Vector3 tADCxcd = -1.f * nADC.crossProduct(z);
      Vector3 tcdxCDB = -1.f * z.crossProduct(nBCD);

      // Edge [C, D] is the VFR
      if ((h.dot(tADCxcd) >= 0.f) && (h.dot(tcdxCDB) >= 0.f) && (g.dot(z) >= 0.f) && (h.dot(z) <= 0.f))
      {
         float t_cd = std::fmin(std::fmax(g.dot(z) / z.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(0.f, 0.f, 1.f - t_cd, t_cd);
         closest_point.point = (1.f - t_cd) * c + t_cd * d;
         return closest_point;
      }

      Vector3 tBDAxad = -1.f * nDAB.crossProduct(v);
      Vector3 tadxADC = -1.f * v.crossProduct(nADC);

      // Edge [A, D] is the VFR
      if ((h.dot(tBDAxad) >= 0.f) && (h.dot(tadxADC) >= 0.f) && (e.dot(v) >= 0.f) && (h.dot(v) <= 0.f))
      {
         float t_ad = std::fmin(std::fmax(e.dot(v) / v.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(1.f - t_ad, 0.f, 0.f, t_ad);
         closest_point.point = (1.f - t_ad) * a + t_ad * d;
         return closest_point;
      }

      Vector3 tDCAxac = -1.f * nADC.crossProduct(w);
      Vector3 tacxACB = -1.f * w.crossProduct(nBAC);

      // Edge [A, C] is the VFR
      if ((g.dot(tDCAxac) >= 0.f) && (g.dot(tacxACB) >= 0.f) && (e.dot(w) >= 0.f) && (g.dot(w) <= 0.f))
      {
         float t_ac = std::fmin(std::fmax(e.dot(w) / w.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(1.f - t_ac, 0.f, t_ac, 0.f);
         closest_point.point = (1.f - t_ac) * a + t_ac * c;
         return closest_point;
      }

      Vector3 tCDBxbd = -1.f * nBCD.crossProduct(y);
      Vector3 tbdxBDA = -1.f * y.crossProduct(nDAB);

      // Edge [B, D] is the VFR
      if ((h.dot(tCDBxbd) >= 0.f) && (h.dot(tbdxBDA) >= 0.f) && (f.dot(y) <= 0.f) && (h.dot(y) <= 0.f))
      {
         float t_bd = std::fmin(std::fmax(f.dot(y) / y.magnitudeSquared(), 0.f), 1.f);
         closest_point.bary.Initialize(0.f, 1.f - t_bd, 0.f, t_bd);
         closest_point.point = (1.f - t_bd) * b + t_bd * d;
         return closest_point;
      }

      // Face [A, B, C] is the VFR
      if (e.dot(nBAC) * v.dot(nBAC) < 0.f)
      {
         float t = nBAC.dot(e) / nBAC.magnitudeSquared();
         closest_point.point = q - t * nBAC;
         closest_point.bary = geometry::triangle::baryCoords(
            a, b, c, closest_point.point
         );

         return closest_point;
      }

      // Face [D, B, A] is the VFR
      if (h.dot(nDAB) * (-1.f * z.dot(nDAB)) < 0.f)
      {
         float t = nDAB.dot(f) / nDAB.magnitudeSquared();
         closest_point.point = q - t * nDAB;
         closest_point.bary = geometry::triangle::baryCoords(
            a, b, d, closest_point.point
         );

         closest_point.bary[3] = closest_point.bary[2];
         closest_point.bary[2] = 0.f;

         return closest_point;
      }

      // Face [C, B, D] is the VFR
      if (g.dot(nBCD) * (-1.f * w.dot(nBCD)) < 0.f)
      {
         float t = nBCD.dot(f) / nBCD.magnitudeSquared();
         closest_point.point = q - t * nBCD;
         closest_point.bary = geometry::triangle::baryCoords(
            b, c, d, closest_point.point
         );

         closest_point.bary[3] = closest_point.bary[2];
         closest_point.bary[2] = closest_point.bary[1];
         closest_point.bary[1] = closest_point.bary[0];
         closest_point.bary[0] = 0.f;

         return closest_point;
      }

      // Face [A, C, D] is the VFR
      if (h.dot(nADC) * (-1.f * y.dot(nADC)) < 0.f)
      {
         float t = nADC.dot(e) / nADC.magnitudeSquared();
         closest_point.point = q - t * nADC;
         closest_point.bary = geometry::triangle::baryCoords(
            a, c, d, closest_point.point
         );

         closest_point.bary[3] = closest_point.bary[2];
         closest_point.bary[2] = closest_point.bary[1];
         closest_point.bary[1] = 0.f;

         return closest_point;
      }

      // The entire tetrahedron is the VFR
      closest_point.point = q;
      closest_point.bary = baryCoords(a, b, c, d, q);

      for (int i = 0; i < 4; ++i)
      {
         closest_point.bary[i] = std::fmax(closest_point.bary[i], 0.f);
      }

      return closest_point;
   }

   bool pointInside(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      const Vector3 & q
   )
   {
      Vector4 bary = baryCoords(a, b, c, d, q);

      for (int i = 0; i < 4; ++i)
      {
         if (bary[i] < 0.f)
         {
            return false;
         }
      }

      return true;
   }

   float volume(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d
   )
   {
      Vector3 vda = a - d;
      Vector3 vdb = b - d;
      Vector3 vdc = c - d;
      Matrix33 T(
         vda[0], vdb[0], vdc[0],
         vda[1], vdb[1], vdc[1],
         vda[2], vdb[2], vdc[2]
      );
      float volume = fabs(T.determinant()) / 6.f;

      return volume;
   }

}

}
