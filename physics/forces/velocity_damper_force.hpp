#ifndef VELOCITY_DAMPER_FORCE_HEADER
#define VELOCITY_DAMPER_FORCE_HEADER

#include "ecs_types.hpp"
#include "slingshot_types.hpp"

namespace oy
{

namespace forces
{

namespace velocity_damper
{

   void evaluateForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceVelocityDamper_t & force,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      oy::types::generalizedForce_t & parent_forque,
      oy::types::generalizedForce_t & child_forque
   );

   void evaluateForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceVelocityDamper_t & force,
      const oy::types::rk4Midpoint_t & parent_body,
      const oy::types::rk4Midpoint_t & child_body,
      oy::types::generalizedForce_t & parent_forque,
      oy::types::generalizedForce_t & child_forque
   );

}

}

}

#endif
