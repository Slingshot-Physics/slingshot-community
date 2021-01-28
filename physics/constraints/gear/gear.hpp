#ifndef GEAR_HEADER
#define GEAR_HEADER

#include "allocator.hpp"
#include "slingshot_types.hpp"

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
   );

   float evaluateBaumgarte(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & gear,
      float dt
   );

   float evaluateConstraint(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & gear
   );

   Matrix<12, 1> calculateMatrixJacobian(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & gear
   );

}

}

}

#endif
