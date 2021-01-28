#include "segment.cuh"

namespace cugeom
{

namespace segment
{
   CUDA_CALLABLE float closestParameterToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & q
   )
   {
      cumath::Vector3 bma(b - a);
      float t = (q - a).dot(bma)/bma.magnitudeSquared();

      return fmaxf(fminf(t, 1.f), 0.f);
   }

   CUDA_CALLABLE pointBaryCoord_t closestPointToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & q
   )
   {
      pointBaryCoord_t bary_point;

      float t = closestParameterToPoint(a, b, q);

      bary_point.point = t * b + (1.f - t) * a;
      bary_point.baryCoords.Initialize(1.f - t, t, 0.f);

      return bary_point;
   }

   CUDA_CALLABLE simplex3_t voronoiRegionToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & q
   )
   {
      simplex3_t region;
      float t = closestParameterToPoint(a, b, q);

      bool t_is_zero = (t == 0.f);
      bool t_is_one = (t == 1.f);

      region.numVerts = 1 + !(t_is_zero || t_is_one);
      region.vertIdxs[0] = 0 + t_is_one;
      region.vertIdxs[1] = 1;
      region.verts[0] = b * (1.f * t_is_one) + a * (1.f - t_is_one);
      region.verts[1] = b;

      return region;
   }

}

}
