#include "revolute_motor.hpp"

#include <limits>

namespace oy
{

namespace constraints
{

namespace revolute_motor
{
   void calculateLambdaBounds(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint,
      float dt,
      float & lambda_min,
      float & lambda_max
   )
   {
      (void)parent_id;
      (void)child_id;
      (void)parent_body;
      (void)child_body;
      lambda_min = -1.0f * constraint.maxTorque * dt;
      lambda_max = constraint.maxTorque * dt;
   }

   float evaluateBaumgarte(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint,
      float dt
   )
   {
      (void)parent_id;
      (void)child_id;
      (void)parent_body;
      (void)child_body;
      (void)dt;
      return -1.f * constraint.angularSpeed;
   }

   float evaluateConstraint(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint
   )
   {
      unsigned int num_bodies = 1 + (
         (parent_id != oy::types::null_body_entity) &&
         (child_id != oy::types::null_body_entity)
      );

      float c = -1.0;
      if (num_bodies == 1)
      {
         const oy::types::rigidBody_t & body = \
            (parent_id != oy::types::null_body_entity) ? parent_body : child_body;

         Vector3 axis_W = body.ql2b.conjugateSandwich(
            (parent_id != -1) ? constraint.parentAxis : constraint.childAxis
         );

         c = body.ql2b.conjugateSandwich(body.angVel).dot(axis_W) - constraint.angularSpeed;
      }
      else
      {
         Vector3 parent_axis_W = parent_body.ql2b.conjugateSandwich(constraint.parentAxis);
         Vector3 child_axis_W = child_body.ql2b.conjugateSandwich(constraint.childAxis);
         
         c = parent_body.ql2b.conjugateSandwich(parent_body.angVel).dot(parent_axis_W) - child_body.ql2b.conjugateSandwich(child_body.angVel).dot(child_axis_W) - constraint.angularSpeed;
      }

      return c;
   }

   Matrix<12, 1> calculateMatrixJacobian(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint
   )
   {
      unsigned int num_bodies = 1 + (
         (parent_id != oy::types::null_body_entity) &&
         (child_id != oy::types::null_body_entity)
      );

      Matrix<12, 1> revolve;
      if (num_bodies == 1)
      {
         const oy::types::rigidBody_t & body = \
            (parent_id != oy::types::null_body_entity) ? parent_body : child_body;

         Vector3 axis_W = body.ql2b.conjugateSandwich((parent_id != -1) ? constraint.parentAxis : constraint.childAxis);

         revolve.assignSlice(
            3 + 6 * (child_id != oy::types::null_body_entity), 0, axis_W
         );
      }
      else
      {
         Vector3 parent_axis_W = parent_body.ql2b.conjugateSandwich(
            constraint.parentAxis
         );
         Vector3 child_axis_W = child_body.ql2b.conjugateSandwich(
            constraint.childAxis
         );

         revolve.assignSlice(3, 0, parent_axis_W);
         revolve.assignSlice(9, 0, -1.f * child_axis_W);

         if (parent_stationary)
         {
            for (int i = 0; i < 6; ++i)
            {
               revolve(i + 0, 0) = 0.f;
            }
         }
         else if (child_stationary)
         {
            for (int i = 0; i < 6; ++i)
            {
               revolve(i + 6, 0) = 0.f;
            }
         }
      }

      return revolve;
   }

}

}

}
