#ifndef TRANSFORM_HEADER
#define TRANSFORM_HEADER

#include "geometry_types.hpp"

namespace geometry
{

namespace transform
{
   // Performs a forward transformation on a bound vector in frame P. This
   // results in vector x_P being expressed in W coordinates, x_W.
   // Bound vectors are usually position vectors.
   //    x_W = trans_P_to_W.rotate * trans_P_to_W.scale * x_P + trans_P_to_W.translate
   Vector3 forwardBound(
      const geometry::types::transform_t & trans_P_to_W, const Vector3 & x_P
   );

   // Performs an inverse transformation on a bound vector in frame W. This
   // results in vector x_W being expressed in P coordinates, x_P.
   // Bound vectors are usually position vectors.
   //    x_P = trans_P_to_W.scale.inverse() * trans_P_to_W.rotate.transpose() * (x_W - trans_P_to_W.translate)
   Vector3 inverseBound(
      const geometry::types::transform_t & trans_P_to_W, const Vector3 & x_W
   );

   // Performs a forward transformation on an unbound vector in frame P. This
   // results in vector x_P being expressed in W coordinates, x_W.
   // Unbound vectors are usually time derivatives of position vectors.
   //    x_W = trans_P_to_W.rotate * trans_P_to_W.scale * x_P
   Vector3 forwardUnbound(
      const geometry::types::transform_t & trans_P_to_W, const Vector3 & x_P
   );

   // Performs a forward transformation on an unbound vector in frame W. This
   // results in a vector x_W being expressed in P coordinates, x_P.
   // Unbound vectors are usually time derivatives of position vectors.
   //    x_P = trans_P_to_W.scale.inverse() * trans_P_to_W.rotate.transpose() * x_W
   Vector3 inverseUnbound(
      const geometry::types::transform_t & trans_P_to_W, const Vector3 & x_W
   );

   // Generates a transformation that's equivalent to the composition of the
   // transforms trans_P_to_Q = trans_W_to_Q o trans_P_to_W, which results in
   // transformations of this nature:
   //
   //    x_Q = trans_W_to_Q( trans_P_to_W( x_P) )
   //
   //    x_Q = (R_W_to_Q * S_WQ * R_P_to_W * S_PW) * x_P +
   //          (R_W_to_Q * S_WQ) * r_PW +
   //          r_WQ
   // Where
   //
   //    trans_P_to_W(x_P) = (R_P_to_W * S_PW) * x_P + r_PW
   //
   // and
   //
   //    trans_W_to_Q(x_W) = (R_W_to_Q * S_WQ) * x_W + r_WQ
   //
   geometry::types::transform_t composePToQ(
      const geometry::types::transform_t & trans_P_to_W,
      const geometry::types::transform_t & trans_W_to_Q
   );

   // Performs a forward transformation on a bound vector in frame P. This
   // results in vector x_P being expressed in W coordinates, x_W.
   // Bound vectors are usually position vectors.
   //    x_W = trans_P_to_W.rotate * x_P + trans_P_to_W.translate
   Vector3 forwardBound(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const Vector3 & x_P
   );

   // Performs an inverse transformation on a bound vector in frame W. This
   // results in vector x_W being expressed in P coordinates, x_P.
   // Bound vectors are usually position vectors.
   //    x_P = trans_P_to_W.rotate.transpose() * (x_W - trans_P_to_W.translate)
   Vector3 inverseBound(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const Vector3 & x_W
   );

   // Performs a forward transformation on an unbound vector in frame P. This
   // results in vector x_P being expressed in W coordinates, x_W.
   // Unbound vectors are usually time derivatives of position vectors.
   //    x_W = trans_P_to_W.rotate * x_P
   Vector3 forwardUnbound(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const Vector3 & x_P
   );

   // Performs a forward transformation on an unbound vector in frame W. This
   // results in a vector x_W being expressed in P coordinates, x_P.
   // Unbound vectors are usually time derivatives of position vectors.
   //    x_P = trans_P_to_W.rotate.transpose() * x_W
   Vector3 inverseUnbound(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const Vector3 & x_W
   );

   // Generates a transformation that's equivalent to the composition of the
   // transforms trans_P_to_Q = trans_W_to_Q o trans_P_to_W, which results in
   // transformations of this nature:
   //
   //    x_Q = trans_W_to_Q( trans_P_to_W( x_P) )
   //
   //    x_Q = (R_W_to_Q * R_P_to_W) * x_P +
   //          R_W_to_Q * r_PW + r_WQ
   // Where
   //
   //    trans_P_to_W(x_P) = R_P_to_W * x_P + r_PW
   //
   // and
   //
   //    trans_W_to_Q(x_W) = R_W_to_Q * x_W + r_WQ
   //
   geometry::types::isometricTransform_t composePToQ(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const geometry::types::isometricTransform_t & trans_W_to_Q
   );
}

}

#endif
