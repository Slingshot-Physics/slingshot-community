// Contains functions for calculating support points on a set of supported
// convex shapes. The point of maximum support for any shape is the one that
// maximizes the expression:
//
//    d.dot(point - body.center)
//
// Where 'd' is a direction vector, and 'point' is a point on the shape's
// surface.

#ifndef SUPPORT_FUNCTIONS_HEADER
#define SUPPORT_FUNCTIONS_HEADER

#include "geometry_types.hpp"
#include "vector3.hpp"
#include "transform.hpp"

namespace geometry
{
   // Calculates the support point on a convex polyhedron from a direction.
   // The direction vector does not have to be normalized.
   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::convexPolyhedron_t & body
   );

   // Calculates the support point on a sphere from a direction.
   // The direction vector does not have to be normalized.
   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::shapeSphere_t & body
   );

   // Calculates the support point on a capsule from a direction.
   // The direction vector does not have to be normalized.
   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::shapeCapsule_t & body
   );

   // Calculates the support point on a cylinder from a direction.
   // The direction vector does not have to be normalized.
   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::shapeCylinder_t & body
   );

   // Calculates the support point on a cube from a direction.
   // The direction vector does not have to be normalized.
   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::shapeCube_t & body
   );

   // Generates a matrix that can be used to transform a support direction in
   // world frame into a support direction that can be used in body frame.
   // The shape-based support functions all operate in body frame (the shape's
   // local reference frame) - a support direction in the shape's frame is used
   // to find a support point in the shape's frame. Most calculations provide
   // a support direction in world frame.
   //
   // dot(x_W, search_W) = dot(R_{W/A} * S_{AW} * x_A, search_W)
   //    = transpose(R_{W/A} * S_{AW} * x_A) * search_W
   //    = trans(x_A) * trans(S_{AW}) * trans(R_{W/A}) * search_W
   //    = dot(x_A, (trans(S_{AW}) * trans(R_{W/A}) * search_W))
   //
   //    ==> support_dir_mat_A = trans(S_{AW}) * trans(R_{W/A})
   //
   // dot(x_W, search_W) = dot(x_A, support_dir_mat_A * search_W)
   Matrix33 supportDirectionTransform(
      const geometry::types::transform_t & trans_A_to_W
   );

   // Generates a matrix that can be used to transform a support direction in
   // world frame into a support direction that can be used in body frame.
   // The shape-based support functions all operate in body frame (the shape's
   // local reference frame) - a support direction in the shape's frame is used
   // to find a support point in the shape's frame. Most calculations provide
   // a support direction in world frame.
   //
   // dot(x_W, search_W) = dot(R_{W/A} * S_{AW} * x_A, search_W)
   //    = transpose(R_{W/A} * S_{AW} * x_A) * search_W
   //    = trans(x_A) * trans(S_{AW}) * trans(R_{W/A}) * search_W
   //    = dot(x_A, (trans(S_{AW}) * trans(R_{W/A}) * search_W))
   //
   //    ==> support_dir_mat_A = trans(S_{AW}) * trans(R_{W/A})
   //
   // dot(x_W, search_W) = dot(x_A, support_dir_mat_A * search_W)
   Matrix33 supportDirectionTransform(
      const geometry::types::isometricTransform_t & trans_A_to_W
   );

   template <typename Shape_T>
   class ShapeSupport
   {
      typedef geometry::types::labeledVertex_t labeledVertex_t;

      public:
         ShapeSupport(
            const geometry::types::transform_t & trans_B_to_W,
            const Shape_T & shape
         )
            : shape_(shape)
            , rotate_scale_B_to_W_(trans_B_to_W.rotate * trans_B_to_W.scale)
            , translate_W_(trans_B_to_W.translate)
            , support_transform_W_to_B_(
               supportDirectionTransform(trans_B_to_W)
            )
         { }

         ShapeSupport(
            const geometry::types::isometricTransform_t & trans_B_to_W,
            const Shape_T & shape
         )
            : shape_(shape)
            , rotate_scale_B_to_W_(trans_B_to_W.rotate)
            , translate_W_(trans_B_to_W.translate)
            , support_transform_W_to_B_(
               supportDirectionTransform(trans_B_to_W)
            )
         { }

         // Calculates the support vertex in bound world coordinates from a
         // search direction specified in world coordinates.
         labeledVertex_t boundSupportWorldDir(const Vector3 & search_dir_W) const
         {
            const labeledVertex_t support_B = geometry::supportMapping(
               support_transform_W_to_B_ * search_dir_W, shape_
            );

            const labeledVertex_t support_W = {
               support_B.vertId,
               rotate_scale_B_to_W_ * support_B.vert + translate_W_
            };

            return support_W;
         }

         // Calculates the support vertex in unbound world coordinates from a
         // search direction specified in world coordinates.
         // Unbound world coordinates is the coordinate system with the shape's
         // center of mass at the origin.
         labeledVertex_t unboundSupportWorldDir(const Vector3 & search_dir_W) const
         {
            const labeledVertex_t support_B = geometry::supportMapping(
               support_transform_W_to_B_ * search_dir_W, shape_
            );

            const labeledVertex_t support_W = {
               support_B.vertId,
               rotate_scale_B_to_W_ * support_B.vert
            };

            return support_W;
         }

         // Calculates the support vertex in unbound world coordinates from a
         // search direction specified in body coordinates.
         labeledVertex_t boundSupportBodyDir(const Vector3 & search_dir_B) const
         {
            const labeledVertex_t support_B = geometry::supportMapping(
               search_dir_B, shape_
            );

            const labeledVertex_t support_W = {
               support_B.vertId,
               rotate_scale_B_to_W_ * support_B.vert + translate_W_
            };

            return support_W;
         }

         // Calculates the support vertex in unbound world coordinates from a
         // search direction specified in body coordinates.
         // Unbound world coordinates is the coordinate system with the shape's
         // center of mass at the origin.
         labeledVertex_t unboundSupportBodyDir(const Vector3 & search_dir_B) const
         {
            const labeledVertex_t support_B = geometry::supportMapping(
               search_dir_B, shape_
            );

            const labeledVertex_t support_W = {
               support_B.vertId,
               rotate_scale_B_to_W_ * support_B.vert
            };

            return support_W;
         }

      private:
         const Shape_T shape_;

         const Matrix33 rotate_scale_B_to_W_;

         const Vector3 translate_W_;

         const Matrix33 support_transform_W_to_B_;
   };

   // This is a helper class that uses shape definitions and transforms to
   // calculate a support point on a Minkowski Difference polytope given a
   // search direction in Minkowski Difference space.
   template <typename Transform_T, typename ShapeA_T, typename ShapeB_T>
   class MinkowskiDiffSupport
   {
      typedef geometry::types::minkowskiDiffVertex_t mdVertex_t;
      typedef geometry::types::labeledVertex_t labeledVertex_t;

      Transform_T trans_A_to_W_;
      Transform_T trans_B_to_W_;
      ShapeA_T shape_a_A_;
      ShapeB_T shape_b_B_;

      // These matrices are linked to the calculation of support points for a
      // shape that is parameterized in collider coordinates (its local
      // unrotated, unscaled, untranslated coordinate frame).
      Matrix33 support_dir_mat_A_;
      Matrix33 support_dir_mat_B_;

      Vector3 transformBound_A_to_W(const Vector3 & v_A)
      {
         return geometry::transform::forwardBound(trans_A_to_W_, v_A);
      }

      Vector3 transformBound_B_to_W(const Vector3 & v_B)
      {
         return geometry::transform::forwardBound(trans_B_to_W_, v_B);
      }

      public:
         MinkowskiDiffSupport(
            const Transform_T & trans_A_to_W,
            const Transform_T & trans_B_to_W,
            const ShapeA_T & shape_a_A,
            const ShapeB_T & shape_b_B
         )
            : trans_A_to_W_(trans_A_to_W)
            , trans_B_to_W_(trans_B_to_W)
            , shape_a_A_(shape_a_A)
            , shape_b_B_(shape_b_B)
            , support_dir_mat_A_(
               supportDirectionTransform(trans_A_to_W)
            )
            , support_dir_mat_B_(
               supportDirectionTransform(trans_B_to_W)
            )
         { }

         // Returns a support point that expands away from the origin in
         // Minkowski difference space.
         mdVertex_t operator()(const Vector3 & search_dir_W)
         {
            const labeledVertex_t support_a_A = geometry::supportMapping(
               support_dir_mat_A_ * search_dir_W, shape_a_A_
            );

            const labeledVertex_t support_b_B = geometry::supportMapping(
               -1.0f * support_dir_mat_B_ * search_dir_W, shape_b_B_
            );

            const Vector3 support_a_W = transformBound_A_to_W(support_a_A.vert);
            const Vector3 support_b_W = transformBound_B_to_W(support_b_B.vert);

            const Vector3 support_md_W = support_a_W - support_b_W;

            mdVertex_t md_vert = {
               support_a_A.vertId, support_b_B.vertId, support_md_W, support_a_A.vert, support_b_B.vert
            };

            return md_vert;
         }
   };

}

#endif
