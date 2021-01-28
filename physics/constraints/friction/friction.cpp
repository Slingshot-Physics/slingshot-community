#include "friction.hpp"

#include <algorithm>

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

      lambda_min = -1.0f * fric.muTotal * abs(normal_force_W);
      lambda_max = fric.muTotal * abs(normal_force_W);
   }

   float evaluateBaumgarte(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintFriction_t & fric,
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
      const oy::types::constraintFriction_t & fric
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
      const oy::types::constraintFriction_t & fric
   )
   {
      (void)body_id_a;
      (void)body_id_b;
      const Vector3 & contact_normal_W = fric.unitNormal;

      // Contact position relative to each body's CM position.
      Vector3 contact_a_rel_cm_W = fric.bodyAContact - body_a.linPos;
      Vector3 contact_b_rel_cm_W = fric.bodyBContact - body_b.linPos;

      Vector3 ang_vel_a_W = body_a.ql2b.conjugateSandwich(body_a.angVel);
      Vector3 ang_vel_b_W = body_b.ql2b.conjugateSandwich(body_b.angVel);

      Vector3 vel_a_rel_b_W = (
         (body_a.linVel + ang_vel_a_W.crossProduct(contact_a_rel_cm_W)) - (body_b.linVel + ang_vel_b_W.crossProduct(contact_b_rel_cm_W))
      );

      Vector3 velARelBUnit = vel_a_rel_b_W.unitVector();

      // This Jacobian implies that the generalized velocity vector is:
      // [ vel_A, angvel_A, vel_B, angvel_B ]
      float relVelDotNormal = velARelBUnit.dot(contact_normal_W);
      Matrix<12, 1> smack;
      smack.assignSlice(
         0, 0, velARelBUnit - relVelDotNormal * contact_normal_W
      );
      smack.assignSlice(
         3, 0, contact_a_rel_cm_W.crossProduct(velARelBUnit) - relVelDotNormal * contact_a_rel_cm_W.crossProduct(contact_normal_W)
      );
      smack.assignSlice(
         6, 0, -1.0f * velARelBUnit + relVelDotNormal * contact_normal_W
      );
      smack.assignSlice(
         9, 0, -1.0f * contact_b_rel_cm_W.crossProduct(velARelBUnit) + relVelDotNormal * contact_b_rel_cm_W.crossProduct(contact_normal_W)
      );

      if (body_a_stationary)
      {
         for (int i = 0; i < 6; ++i)
         {
            smack(i + 0, 0) = 0.f;
         }
      }
      else if (body_b_stationary)
      {
         for (int i = 0; i < 6; ++i)
         {
            smack(i + 6, 0) = 0.f;
         }
      }

      return smack;
   }

}

}

}
