#ifndef DRAG_TORQUE_HEADER
#define DRAG_TORQUE_HEADER

#include "slingshot_types.hpp"

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
   );

   void evaluateTorque(
      const oy::types::torqueDrag_t & torque,
      const oy::types::rk4Midpoint_t & body,
      oy::types::generalizedForce_t & forque
   );

}

}

}

#endif
