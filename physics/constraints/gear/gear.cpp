#include "gear.hpp"

#include <algorithm>
#include <limits>

namespace oy
{

namespace constraints
{

namespace gear
{

   void calculateLambdaBounds(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & gear,
      float & lambda_min,
      float & lambda_max
   )
   {
      (void)parent_id;
      (void)child_id;
      (void)parent_body;
      (void)child_body;
      (void)gear;
      lambda_min = -1.0f * std::numeric_limits<float>::infinity();
      lambda_max = std::numeric_limits<float>::infinity();
   }

   float evaluateBaumgarte(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & gear,
      float dt
   )
   {
      (void)parent_id;
      (void)child_id;
      (void)parent_body;
      (void)child_body;
      (void)gear;
      (void)dt;
      return 0.0f;
   }

   float evaluateConstraint(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & gear
   )
   {
      (void)parent_id;
      (void)child_id;
      (void)parent_body;
      (void)child_body;
      (void)gear;
      return 0.0f;
   }

   Matrix<12, 1> calculateMatrixJacobian(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & gear
   )
   {
      (void)parent_id;
      (void)child_id;

      Vector3 parent_axis_W = parent_body.ql2b.conjugateSandwich(
         gear.parentAxis
      );
      Vector3 child_axis_W = child_body.ql2b.conjugateSandwich(
         gear.childAxis
      );

      // This Jacobian implies that the generalized velocity vector is:
      // [ vel_parent, angvel_parent, vel_child, angvel_child ]
      Matrix<12, 1> crank;
      crank.assignSlice(3, 0, gear.parentGearRadius * parent_axis_W);
      float rotation_sign = 1.f - 2.f * gear.rotateParallel;
      crank.assignSlice(9, 0, rotation_sign * gear.childGearRadius * child_axis_W);

      if (parent_stationary)
      {
         for (int i = 0; i < 6; ++i)
         {
            crank(i + 0, 0) = 0.f;
         }
      }
      else if (child_stationary)
      {
         for (int i = 0; i < 6; ++i)
         {
            crank(i + 6, 0) = 0.f;
         }
      }
      return crank;
   }

}

}

}
