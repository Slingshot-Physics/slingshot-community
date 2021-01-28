#include "drag_force.hpp"

#include "rigidbody.hpp"

namespace oy
{

namespace forces
{

namespace drag
{

   void evaluateForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceDrag_t & force_config,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      oy::types::generalizedForce_t & parent_forque,
      oy::types::generalizedForce_t & child_forque
   )
   {
      (void)parent_id;
      (void)child_id;
      (void)parent_body;
      (void)parent_forque;

      oy::rb::applyForce(
         child_body,
         force_config.linearDragCoeff * child_body.linVel + force_config.quadraticDragCoeff * child_body.linVel * child_body.linVel.magnitude(),
         oy::types::enumFrame_t::GLOBAL,
         child_forque
      );
   }

   void evaluateForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceDrag_t & force_config,
      const oy::types::rk4Midpoint_t & parent_body,
      const oy::types::rk4Midpoint_t & child_body,
      oy::types::generalizedForce_t & parent_forque,
      oy::types::generalizedForce_t & child_forque
   )
   {
      (void)parent_id;
      (void)child_id;
      (void)parent_body;
      (void)parent_forque;

      oy::rb::applyForce(
         force_config.linearDragCoeff * child_body.linVel + force_config.quadraticDragCoeff * child_body.linVel * child_body.linVel.magnitude(),
         oy::types::enumFrame_t::GLOBAL,
         child_body,
         child_forque
      );
   }

}

}

}
