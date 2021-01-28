#include "constant_force.hpp"

#include "rigidbody.hpp"

namespace oy
{

namespace forces
{

namespace constant
{

   void evaluateForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceConstant_t & force_config,
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

      Vector3 child_link_point = force_config.childLinkPoint;

      if (force_config.forceFrame == oy::types::enumFrame_t::GLOBAL)
      {
         child_link_point = child_body.ql2b.conjugateSandwich(child_link_point);
      }

      oy::rb::applyForce(
         child_body,
         child_body.mass * force_config.acceleration,
         child_link_point,
         force_config.forceFrame,
         child_forque
      );
   }

   void evaluateForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceConstant_t & force_config,
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

      Vector3 child_link_point = force_config.childLinkPoint;

      if (force_config.forceFrame == oy::types::enumFrame_t::GLOBAL)
      {
         child_link_point = child_body.ql2b.conjugateSandwich(child_link_point);
      }

      oy::rb::applyForce(
         child_body.mass * force_config.acceleration,
         child_link_point,
         force_config.forceFrame,
         child_body,
         child_forque
      );
   }
}

}

}
