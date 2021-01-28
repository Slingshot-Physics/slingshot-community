#include "rotation_1d.hpp"

#include <limits>

namespace oy
{

namespace constraints
{

namespace rotation_1d
{
   void calculateLambdaBounds(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      float & lambda_min,
      float & lambda_max
   )
   {
      (void)parent_id;
      (void)child_id;
      (void)parent_body;
      (void)child_body;
      lambda_min = -1.0f * std::numeric_limits<float>::infinity();
      lambda_max = std::numeric_limits<float>::infinity();
   }

   float evaluateBaumgarte(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRotation1d_t & constraint,
      float dt
   )
   {
      (void)parent_id;
      (void)child_id;
      return 0.2f * evaluateConstraint(
         parent_id, child_id, parent_body, child_body, constraint
      ) / dt;
   }

   float evaluateConstraint(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRotation1d_t & constraint
   )
   {
      (void)parent_id;
      (void)child_id;
      const Vector3 parent_axis_W = parent_body.ql2b.conjugateSandwich(
         constraint.parentAxis
      );

      const Vector3 child_axis_W = child_body.ql2b.conjugateSandwich(
         constraint.childAxis
      );

      const float c = (parent_axis_W).dot(child_axis_W);
      return c;
   }

   Matrix<12, 1> calculateMatrixJacobian(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRotation1d_t & constraint
   )
   {
      (void)parent_id;
      (void)child_id;

      const Vector3 parent_axis_W = parent_body.ql2b.conjugateSandwich(
         constraint.parentAxis
      ).Normalize();

      const Vector3 child_axis_W = child_body.ql2b.conjugateSandwich(
         constraint.childAxis
      ).Normalize();

      const Vector3 parent_cross_child_W = parent_axis_W.crossProduct(child_axis_W);

      Matrix<12, 1> swivel;
      swivel.assignSlice(
         3, 0, parent_cross_child_W
      );
      swivel.assignSlice(
         9, 0, -1.f * parent_cross_child_W
      );

      if (parent_stationary)
      {
         for (int i = 0; i < 6; ++i)
         {
            swivel(i + 0, 0) = 0.f;
         }
      }
      else if (child_stationary)
      {
         for (int i = 0; i < 6; ++i)
         {
            swivel(i + 6, 0) = 0.f;
         }
      }

      return swivel;
   }

}

}

}
