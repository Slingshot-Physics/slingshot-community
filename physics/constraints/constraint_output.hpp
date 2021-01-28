#ifndef CONSTRAINT_OUTPUT_HEADER
#define CONSTRAINT_OUTPUT_HEADER

#include "ecs_types.hpp"
#include "slingshot_types.hpp"

namespace oy
{
   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintCollision_t & constraint,
      float dt
   );

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintFriction_t & constraint,
      float dt
   );

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & constraint,
      float dt
   );

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint,
      float dt
   );

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRotation1d_t & constraint,
      float dt
   );

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintTorsionalFriction_t & constraint,
      float dt
   );

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintTranslation1d_t & constraint,
      float dt
   );

}

#endif
