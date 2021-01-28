#ifndef COLLISION_HEADER
#define COLLISION_HEADER

#include "allocator.hpp"
#include "slingshot_types.hpp"

namespace oy
{

namespace constraints
{

namespace collision
{
   // Puts bounds on the values of lambda
   void calculateLambdaBounds(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      float & lambda_min,
      float & lambda_max
   );

   // Returns a Baumgarte stabilization factor for the collision constraint.
   float evaluateBaumgarte(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintCollision_t & collision,
      float dt
   );

   float evaluateConstraint(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintCollision_t & collision
   );

   Matrix<12, 1> calculateMatrixJacobian(
      bool body_a_stationary,
      bool body_b_stationary,
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintCollision_t & collision
   );

}

}

}

#endif
