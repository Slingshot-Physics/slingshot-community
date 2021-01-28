#ifndef TETRAHEDRON_HEADER
#define TETRAHEDRON_HEADER

#include "geometry_types.hpp"

namespace geometry
{

namespace tetrahedron
{
   typedef geometry::types::pointBaryCoord_t pointBaryCoord_t;
   typedef geometry::types::tetrahedron_t tetrahedron_t;

   // Vertices A and B are the edge vertices. C and D are the remaining
   // vertices.
   const struct edge_s {
      int vertAId;
      int vertBId;
      int vertCId;
      int vertDId;
   } edgeCombos[6] = {
      {0, 1, 2, 3},
      {0, 2, 1, 3},
      {0, 3, 1, 2},
      {1, 2, 0, 3},
      {1, 3, 0, 2},
      {2, 3, 0, 1}
   };

   // First three vertices form the face.
   const struct face_s
   {
      int vertAId;
      int vertBId;
      int vertCId;
      int vertDId;
   } faceCombos[4] = {
      {0, 1, 2, 3},
      {0, 3, 1, 2},
      {1, 2, 3, 0},
      {2, 3, 0, 1}
   };

   // Calculate the barycentric coordinates of the query point 'q' given a
   // tetrahedron definition.
   Vector4 baryCoords(const tetrahedron_t & tetra, const Vector3 & q);

   // Calculate the barycentric coordinates of the query point 'q' given a
   // tetrahedron definition.
   Vector4 baryCoords(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      const Vector3 & q
   );

   pointBaryCoord_t closestPointToPoint(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      const Vector3 & q
   );

   bool pointInside(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d,
      const Vector3 & q
   );

   float volume(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & d
   );

}

}

#endif
