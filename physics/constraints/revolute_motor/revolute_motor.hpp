#ifndef REVOLUTE_MOTOR_CONSTRAINT_HEADER
#define REVOLUTE_MOTOR_CONSTRAINT_HEADER

#include "allocator.hpp"
#include "slingshot_types.hpp"

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
   );

   // Returns a Baumgarte stabilization factor for the revolute joint constraint.
   float evaluateBaumgarte(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint,
      float dt
   );

   // Returns the value of the constraint evaluated at the constrained bodies'
   // current configuration.
   float evaluateConstraint(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint
   );

   // Lets two bodies be linked by one revolute joint.
   Matrix<12, 1> calculateMatrixJacobian(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint
   );

}

}

}

#endif
