#include "torsional_friction.hpp"

#include <algorithm>

namespace oy
{

namespace constraints
{

namespace torsional_friction
{
   void calculateLambdaBounds(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintTorsionalFriction_t & fric,
      float & lambda_min,
      float & lambda_max
   )
   {
      (void)body_id_a;
      (void)body_id_b;
      (void)body_a;
      (void)body_b;

      float normal_force_a_W = fric.unitNormal.dot(fric.bodyAForce);
      float normal_force_b_W = -1.f * fric.unitNormal.dot(fric.bodyBForce);

      float normal_force_W = std::max(normal_force_a_W, normal_force_b_W);
      normal_force_W = std::max(0.f, normal_force_W);

      lambda_min = -1.0f * fric.muTotal * abs(normal_force_W) * abs(fric.leverArmLength);
      lambda_max = fric.muTotal * abs(normal_force_W)* abs(fric.leverArmLength);
   }

   float evaluateBaumgarte(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintTorsionalFriction_t & fric,
      float dt
   )
   {
      (void)body_id_a;
      (void)body_id_b;
      (void)body_a;
      (void)body_b;
      (void)fric;
      (void)dt;
      return 0.0f;
   }

   float evaluateConstraint(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintTorsionalFriction_t & fric
   )
   {
      (void)body_id_a;
      (void)body_id_b;
      (void)body_a;
      (void)body_b;
      (void)fric;
      return 0.0f;
   }

   Matrix<12, 1> calculateMatrixJacobian(
      bool body_a_stationary,
      bool body_b_stationary,
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintTorsionalFriction_t & fric
   )
   {
      (void)body_id_a;
      (void)body_id_b;
      (void)body_a;
      (void)body_b;

      Matrix<12, 1> spin;
      spin.assignSlice(3, 0, fric.unitNormal);
      spin.assignSlice(9, 0, -1.f * fric.unitNormal);

      if (body_a_stationary)
      {
         for (int i = 0; i < 6; ++i)
         {
            spin(i + 0, 0) = 0.f;
         }
      }
      else if (body_b_stationary)
      {
         for (int i = 0; i < 6; ++i)
         {
            spin(i + 6, 0) = 0.f;
         }
      }

      return spin;
   }

}

}

}
