#ifndef FRICTION_HEADER
#define FRICTION_HEADER

#include "allocator.hpp"
#include "slingshot_types.hpp"

#include <vector>

namespace oy
{

namespace constraints
{

namespace friction
{
   void calculateLambdaBounds(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintFriction_t & fric,
      float & lambda_min,
      float & lambda_max
   );

   float evaluateBaumgarte(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintFriction_t & fric,
      float dt
   );

   float evaluateConstraint(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintFriction_t & fric
   );

   Matrix<12, 1> calculateMatrixJacobian(
      bool body_a_stationary,
      bool body_b_stationary,
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintFriction_t & fric
   );

}

}

}

#endif
