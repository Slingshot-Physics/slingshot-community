#include "drag_torque.hpp"

#include "rigidbody.hpp"

namespace oy
{

namespace torques
{

namespace drag
{

   void evaluateTorque(
      const oy::types::torqueDrag_t & torque,
      const oy::types::rigidBody_t & body,
      oy::types::generalizedForce_t & forque
   )
   {
      const Vector3 torque_B = (
         torque.linearDragCoeff * body.angVel + torque.quadraticDragCoeff * body.angVel * body.angVel.magnitude()
      );

      oy::rb::applyTorque(
         body,
         torque_B,
         oy::types::enumFrame_t::BODY,
         forque
      );
   }

   void evaluateTorque(
      const oy::types::torqueDrag_t & torque,
      const oy::types::rk4Midpoint_t & body,
      oy::types::generalizedForce_t & forque
   )
   {
      const Vector3 torque_B = (
         torque.linearDragCoeff * body.angVel + torque.quadraticDragCoeff * body.angVel * body.angVel.magnitude()
      );

      oy::rb::applyTorque(
         torque_B,
         oy::types::enumFrame_t::BODY,
         body,
         forque
      );
   }

}

}

}
