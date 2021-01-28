#ifndef GEOMETRY_TYPES_CUDA_HEADER
#define GEOMETRY_TYPES_CUDA_HEADER

#include "vector3.cuh"

namespace cugeom
{

namespace types
{

   struct pointBaryCoord_t
   {
      cumath::Vector3 point;
      cumath::Vector3 baryCoords;
   };

   struct simplex3_t
   {
      int numVerts;
      int vertIdxs[4];
      cumath::Vector3 verts[4];
   };

}

}

#endif
