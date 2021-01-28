#include "transform.hpp"

namespace geometry
{

namespace transform
{

   Vector3 forwardBound(
      const geometry::types::transform_t & trans_P_to_W, const Vector3 & x_P
   )
   {
      return (
         trans_P_to_W.rotate * trans_P_to_W.scale * x_P + trans_P_to_W.translate
      );
   }

   Vector3 inverseBound(
      const geometry::types::transform_t & trans_P_to_W,
      const Vector3 & x_W
   )
   {
      return (
         (~trans_P_to_W.scale) * (
            trans_P_to_W.rotate.transpose() * (x_W - trans_P_to_W.translate)
         )
      );
   }

   Vector3 forwardUnbound(
      const geometry::types::transform_t & trans_P_to_W, const Vector3 & x_P
   )
   {
      return trans_P_to_W.rotate * trans_P_to_W.scale * x_P;
   }

   Vector3 inverseUnbound(
      const geometry::types::transform_t & trans_P_to_W, const Vector3 & x_W
   )
   {
      return (
         (~trans_P_to_W.scale) * (trans_P_to_W.rotate.transpose() * x_W)
      );
   }

   geometry::types::transform_t composePToQ(
      const geometry::types::transform_t & trans_P_to_W,
      const geometry::types::transform_t & trans_W_to_Q
   )
   {
      Matrix33 R_P_to_Q(trans_W_to_Q.rotate * trans_P_to_W.rotate);

      geometry::types::transform_t trans_P_to_Q = {
         trans_P_to_W.rotate.transpose() * trans_W_to_Q.scale * (
            trans_P_to_W.rotate * trans_P_to_W.scale
         ),
         R_P_to_Q,
         (trans_W_to_Q.rotate * (trans_W_to_Q.scale * trans_P_to_W.translate)) +
         trans_W_to_Q.translate
      };

      return trans_P_to_Q;
   }
///

   Vector3 forwardBound(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const Vector3 & x_P
   )
   {
      return trans_P_to_W.rotate * x_P + trans_P_to_W.translate;
   }

   Vector3 inverseBound(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const Vector3 & x_W
   )
   {
      return trans_P_to_W.rotate.transpose() * (x_W - trans_P_to_W.translate);
   }

   Vector3 forwardUnbound(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const Vector3 & x_P
   )
   {
      return trans_P_to_W.rotate * x_P;
   }

   Vector3 inverseUnbound(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const Vector3 & x_W
   )
   {
      return trans_P_to_W.rotate.transpose() * x_W;
   }

   geometry::types::isometricTransform_t composePToQ(
      const geometry::types::isometricTransform_t & trans_P_to_W,
      const geometry::types::isometricTransform_t & trans_W_to_Q
   )
   {
      geometry::types::isometricTransform_t trans_P_to_Q = {
         trans_W_to_Q.rotate * trans_P_to_W.rotate,
         (trans_W_to_Q.rotate * trans_P_to_W.translate) + trans_W_to_Q.translate
      };

      return trans_P_to_Q;
   }
}

}
