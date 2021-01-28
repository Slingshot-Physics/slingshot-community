#include "simplex.cuh"

#include "segment.cuh"
#include "triangle.cuh"
#include "tetrahedron.cuh"

namespace cugeom
{

namespace simplex
{

   CUDA_CALLABLE simplex3_t voronoiRegionToPoint(
      const simplex3_t & simplex, const cumath::Vector3 & q
   )
   {
      simplex3_t min_simplex;

      switch(simplex.numVerts)
      {
         case 1:
         {
            min_simplex.numVerts = 1;
            min_simplex.vertIdxs[0] = 0;
            min_simplex.verts[0] = simplex.verts[0];
            break;
         }
         case 2:
         {
            min_simplex = cugeom::segment::voronoiRegionToPoint(
               simplex.verts[0], simplex.verts[1], q
            );
            break;
         }
         case 3:
         {
            min_simplex = cugeom::triangle::voronoiRegionToPoint(
               simplex.verts[0], simplex.verts[1], simplex.verts[2], q
            );
            break;
         }
         case 4:
         {
            min_simplex = cugeom::tetrahedron::voronoiRegionToPoint(
               simplex.verts[0],
               simplex.verts[1],
               simplex.verts[2],
               simplex.verts[3],
               q
            );
            break;
         }
         default:
            break;
      }

      return min_simplex;
   }

   CUDA_CALLABLE pointBaryCoord_t closestPointToPoint(
      const simplex3_t & simplex, const cumath::Vector3 & q
   )
   {
      pointBaryCoord_t bary_point;

      switch(simplex.numVerts)
      {
         case 1:
         {
            bary_point.point = simplex.verts[0];
            bary_point.baryCoords.Initialize(1.f, 0.f, 0.f);
            break;
         }
         case 2:
         {
            bary_point = cugeom::segment::closestPointToPoint(
               simplex.verts[0], simplex.verts[1], q
            );
            break;
         }
         case 3:
         {
            bary_point = cugeom::triangle::closestPointToPoint(
               simplex.verts[0], simplex.verts[1], simplex.verts[2], q
            );
            break;
         }
         case 4:
         {
            bary_point = cugeom::tetrahedron::closestPointToPoint(
               simplex.verts[0],
               simplex.verts[1],
               simplex.verts[2],
               simplex.verts[3],
               q
            );
            break;
         }
         default:
            break;
      }

      return bary_point;
   }

}

}
