#include <algorithm>
#include <cmath>

#include "mesh_ops.hpp"

namespace geometry
{

   void calcNormal(
      const Vector3 * verts, const Vector3 & center, Vector3 & normal
   )
   {
      calcNormal(verts[0], verts[1], verts[2], center, normal);
   }

   Vector3 calcNormal(const Vector3 * verts, const Vector3 & center)
   {
      Vector3 n;
      calcNormal(verts[0], verts[1], verts[2], center, n);
      return n;
   }

   Vector3 calcNormal(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & center
   )
   {
      Vector3 normal;
      calcNormal(a, b, c, center, normal);
      return normal;
   }

   void calcNormal(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & center,
      Vector3 & normal
   )
   {
      normal = (b - a).crossProduct(c - a);
      normal *= (((a - center).dot(normal) >= 0.0f) ? 1.0f : -1.0f);
   }

   void averageVertex(
      unsigned int numVerts, const Vector3 * verts, Vector3 & center
   )
   {
      for (unsigned int i = 0; i < numVerts; ++i)
      {
         center += verts[i];
      }
      center /= numVerts;
   }

   Vector3 averageVertex(unsigned int numVerts, const Vector3 * verts)
   {
      Vector3 result;
      averageVertex(numVerts, verts, result);
      return result;
   }

}
