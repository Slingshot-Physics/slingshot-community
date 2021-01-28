#ifndef MESH_OPS_HEADER
#define MESH_OPS_HEADER

#include "vector3.hpp"

namespace geometry
{

   // Accepts an array of three vertices. Returns the normal.
   Vector3 calcNormal(const Vector3 * verts, const Vector3 & center);

   // Accepts an array of three vertices. Sets the normal as an output arg.
   void calcNormal(
      const Vector3 * verts, const Vector3 & center, Vector3 & normal
   );

   Vector3 calcNormal(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & center
   );

   void calcNormal(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & center,
      Vector3 & normal
   );

   // Calculates the average vertex of an array of vertices.
   void averageVertex(
      unsigned int numVerts, const Vector3 * verts, Vector3 & center
   );

   // Calculates the average vertex of an array of vertices.
   Vector3 averageVertex(unsigned int numVerts, const Vector3 * verts);

}

#endif
