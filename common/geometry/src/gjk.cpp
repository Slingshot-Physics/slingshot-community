#include "geometry.hpp"
#include "gjk.hpp"

#include <cassert>

namespace geometry
{

namespace gjk
{
   // Adds a vertex to a simplex with a maximum of 4 points. The simplex tracks
   // the contributing vertices and their IDs from the bodies under
   // consideration.
   void addVertexToSimplex(
      const gjk_mdVertex_t & md_vert, gjk_simplex_t & simplex
   )
   {
      assert(simplex.numVerts > 0 && simplex.numVerts <= 4);

      simplex.bodyAVertIds[simplex.numVerts] = md_vert.bodyAVertId;
      simplex.bodyBVertIds[simplex.numVerts] = md_vert.bodyBVertId;
      simplex.bodyAVerts[simplex.numVerts] = md_vert.bodyAVert;
      simplex.bodyBVerts[simplex.numVerts] = md_vert.bodyBVert;
      simplex.verts[simplex.numVerts] = md_vert.vert;
      simplex.numVerts += 1;
   }

   bool pointCoincident(const Vector3 & vert, const gjk_simplex_t & simplex)
   {
      switch(simplex.numVerts)
      {
         case 1:
         {
            return geometry::segment::length(simplex.verts[0], vert) <= 1e-6f;
         }
         case 2:
         {
            return geometry::triangle::area(
               simplex.verts[0], simplex.verts[1], vert
            ) <= 1e-6f;
         }
         case 3:
         {
            return geometry::tetrahedron::volume(
               simplex.verts[0], simplex.verts[1], simplex.verts[2], vert
            ) <= 1e-6f;
         }
         default:
            break;
      }

      return false;
   }

   geometry::types::pointBaryCoord_t minNormPoint(
      const gjk_simplex_t & simplex
   )
   {
      Vector3 zero;
      geometry::types::pointBaryCoord_t closest_point;
      switch(simplex.numVerts)
      {
         case 1:
         {
            closest_point.point = simplex.verts[0];
            closest_point.bary.Initialize(1.f, 0.f, 0.f, 0.f);
            break;
         }
         case 2:
         {
            closest_point = geometry::segment::closestPointToPoint(
               simplex.verts[0], simplex.verts[1], zero
            );
            break;
         }
         case 3:
         {
            closest_point = geometry::triangle::closestPointToPoint(
               simplex.verts[0], simplex.verts[1], simplex.verts[2], zero
            );
            break;
         }
         case 4:
         {
            closest_point = geometry::tetrahedron::closestPointToPoint(
               simplex.verts[0],
               simplex.verts[1],
               simplex.verts[2],
               simplex.verts[3],
               zero
            );
            break;
         }
         default:
            break;
      }

      return closest_point;
   }

   gjk_simplex_t minSimplex(
      const Vector4 & min_norm_bary, const gjk_simplex_t & simplex
   )
   {
      gjk_simplex_t new_simplex;
      unsigned int & num_verts = new_simplex.numVerts;
      num_verts = 0;

      for (unsigned int i = 0; i < 4; ++i)
      {
         if (min_norm_bary[i] > 0.f)
         {
            new_simplex.bodyAVertIds[num_verts] = simplex.bodyAVertIds[i];
            new_simplex.bodyBVertIds[num_verts] = simplex.bodyBVertIds[i];
            new_simplex.bodyAVerts[num_verts] = simplex.bodyAVerts[i];
            new_simplex.bodyBVerts[num_verts] = simplex.bodyBVerts[i];
            new_simplex.verts[num_verts] = simplex.verts[i];
            new_simplex.minNormBary[num_verts] = min_norm_bary[i];
            num_verts += 1;
         }
      }

      return new_simplex;
   }

}

}
