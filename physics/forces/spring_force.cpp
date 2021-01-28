#include "spring_force.hpp"

#include "rigidbody.hpp"

namespace oy
{

namespace forces
{

namespace spring
{

   void evaluateForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceSpring_t & force,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      oy::types::generalizedForce_t & parent_forque,
      oy::types::generalizedForce_t & child_forque
   )
   {
      unsigned int num_bodies = 1 + (
         (parent_id != oy::types::null_body_entity) &&
         (child_id != oy::types::null_body_entity)
      );

      if (num_bodies == 1)
      {
         const oy::types::rigidBody_t & body = \
            (parent_id != oy::types::null_body_entity) ? parent_body : child_body;

         Vector3 linkPointBody_W(
            body.ql2b.conjugateSandwich(
               (parent_id != oy::types::null_body_entity) ? force.parentLinkPoint : force.childLinkPoint
            ) + body.linPos
         );

         Vector3 linkPoint_W(
            (parent_id == oy::types::null_body_entity) ? force.parentLinkPoint : force.childLinkPoint
         );

         Vector3 spring_W(linkPointBody_W - linkPoint_W);
         Vector3 springUnit_W(spring_W.unitVector());
         Vector3 restPoint_W(force.restLength * springUnit_W);

         Vector3 forceAToB_W = force.springCoeff * (spring_W - restPoint_W);

         oy::types::generalizedForce_t & forque = \
            (parent_id != oy::types::null_body_entity) ? parent_forque : child_forque;

         oy::rb::applyForce(
            body,
            forceAToB_W,
            linkPoint_W - body.linPos,
            oy::types::enumFrame_t::GLOBAL,
            forque
         );
      }
      else
      {
         Vector3 linkPointBodyA_W = parent_body.ql2b.conjugateSandwich(
            force.parentLinkPoint
         ) + parent_body.linPos;
         Vector3 linkPointBodyB_W = child_body.ql2b.conjugateSandwich(
            force.childLinkPoint
         ) + child_body.linPos;

         Vector3 spring_W(linkPointBodyB_W - linkPointBodyA_W);
         Vector3 springUnit_W(spring_W.unitVector());
         Vector3 restPoint_W(force.restLength * springUnit_W);

         Vector3 forceAToB_W = force.springCoeff * (spring_W - restPoint_W);

         oy::rb::applyForce(
            child_body,
            forceAToB_W,
            linkPointBodyB_W - child_body.linPos,
            oy::types::enumFrame_t::GLOBAL,
            child_forque
         );
         oy::rb::applyForce(
            parent_body,
            -1.f * forceAToB_W,
            linkPointBodyA_W - parent_body.linPos,
            oy::types::enumFrame_t::GLOBAL,
            parent_forque
         );
      }
   }

   void evaluateForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceSpring_t & force,
      const oy::types::rk4Midpoint_t & parent_body,
      const oy::types::rk4Midpoint_t & child_body,
      oy::types::generalizedForce_t & parent_forque,
      oy::types::generalizedForce_t & child_forque
   )
   {
      unsigned int num_bodies = 1 + (
         (parent_id != oy::types::null_body_entity) &&
         (child_id != oy::types::null_body_entity)
      );

      if (num_bodies == 1)
      {
         const oy::types::rk4Midpoint_t & body = \
            (parent_id != oy::types::null_body_entity) ? parent_body : child_body;

         oy::types::generalizedForce_t & forque = \
            (parent_id != oy::types::null_body_entity) ? parent_forque : child_forque;

         Vector3 linkPointBody_W(
            body.ql2b.conjugateSandwich(
               (parent_id != oy::types::null_body_entity) ? force.parentLinkPoint : force.childLinkPoint
            ) + body.linPos
         );

         Vector3 linkPoint_W(
            (parent_id == oy::types::null_body_entity) ? force.parentLinkPoint : force.childLinkPoint
         );

         Vector3 spring_W(linkPointBody_W - linkPoint_W);
         Vector3 springUnit_W(spring_W.unitVector());
         Vector3 restPoint_W(force.restLength * springUnit_W);

         Vector3 forceAToB_W = force.springCoeff * (spring_W - restPoint_W);

         oy::rb::applyForce(
            forceAToB_W,
            linkPoint_W - body.linPos,
            oy::types::enumFrame_t::GLOBAL,
            body,
            forque
         );
      }
      else
      {
         Vector3 linkPointBodyA_W = parent_body.ql2b.conjugateSandwich(
            force.parentLinkPoint
         ) + parent_body.linPos;
         Vector3 linkPointBodyB_W = child_body.ql2b.conjugateSandwich(
            force.childLinkPoint
         ) + child_body.linPos;

         Vector3 spring_W(linkPointBodyB_W - linkPointBodyA_W);
         Vector3 springUnit_W(spring_W.unitVector());
         Vector3 restPoint_W(force.restLength * springUnit_W);

         Vector3 forceAToB_W = force.springCoeff * (spring_W - restPoint_W);

         oy::rb::applyForce(
            forceAToB_W,
            linkPointBodyB_W - child_body.linPos,
            oy::types::enumFrame_t::GLOBAL,
            child_body,
            child_forque
         );

         oy::rb::applyForce(
            -1.f * forceAToB_W,
            linkPointBodyA_W - parent_body.linPos,
            oy::types::enumFrame_t::GLOBAL,
            parent_body,
            parent_forque
         );
      }
   }
}

}

}
