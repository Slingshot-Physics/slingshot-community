#include "collision.hpp"

#include <limits>

namespace oy
{

namespace constraints
{

namespace collision
{
   void calculateLambdaBounds(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      float & lambda_min,
      float & lambda_max
   )
   {
      (void)body_id_a;
      (void)body_id_b;
      (void)body_a;
      (void)body_b;
      lambda_min = 0.0f;
      lambda_max = std::numeric_limits<float>::infinity();
   }

   float evaluateBaumgarte(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintCollision_t & collision,
      float dt
   )
   {
      // Maximum penetration depth between two bodies before stabilizing.
      const float slop = -1e-4f;

      Vector3 ang_vel_a_W = body_a.ql2b.conjugateSandwich(body_a.angVel);
      Vector3 ang_vel_b_W = body_b.ql2b.conjugateSandwich(body_b.angVel);

      // The initial velocity for static bodies feeds into the impulse response
      // since the engine doesn't propagate stationary bodies.
      Vector3 vel_b_rel_a_W = (
         body_b.linVel + ang_vel_b_W.crossProduct(collision.bodyBContact - body_b.linPos) - 
         (body_a.linVel + ang_vel_a_W.crossProduct(collision.bodyAContact - body_a.linPos))
      );

      float penetration = evaluateConstraint(body_id_a, body_id_b, body_a, body_b, collision);

      float normal_speed_b_rel_a = vel_b_rel_a_W.dot(collision.unitNormal);

      float baumgarte = 0.0f;
      float restitution = collision.restitution * normal_speed_b_rel_a;
      if (penetration < slop)
      {
         baumgarte = (0.025f) * ((penetration - slop) / dt);
         baumgarte += restitution * (restitution < 0.0f);
      }

      return baumgarte;
   }

   float evaluateConstraint(
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintCollision_t & collision
   )
   {
      (void)body_id_a;
      (void)body_id_b;
      (void)body_a;
      (void)body_b;
      const Vector3 & contact_normal_W = collision.unitNormal;

      // Contact position on body A in world coordinates.
      Vector3 contact_a_W = collision.bodyAContact;

      // Contact position on body B in world coordinates.
      Vector3 contact_b_W = collision.bodyBContact;

      float c = (contact_b_W - contact_a_W).dot(contact_normal_W);

      return c;
   }

   Matrix<12, 1> calculateMatrixJacobian(
      bool body_a_stationary,
      bool body_b_stationary,
      trecs::uid_t body_id_a,
      trecs::uid_t body_id_b,
      const oy::types::rigidBody_t & body_a,
      const oy::types::rigidBody_t & body_b,
      const oy::types::constraintCollision_t & collision
   )
   {
      (void)body_id_a;
      (void)body_id_b;
      const Vector3 & contact_normal_W = collision.unitNormal;

      // Contact position relative to each body's CM position.
      Vector3 contact_a_rel_cm_W = collision.bodyAContact - body_a.linPos;
      Vector3 contact_b_rel_cm_W = collision.bodyBContact - body_b.linPos;

      // This Jacobian implies that the generalized velocity vector is:
      // [ vel_A, angvel_A, vel_B, angvel_B ]
      Matrix<12, 1> smack;
      smack.assignSlice(0, 0, -1.0f * contact_normal_W);
      smack.assignSlice(3, 0, -1.0f * contact_a_rel_cm_W.crossProduct(contact_normal_W));
      smack.assignSlice(6, 0, contact_normal_W);
      smack.assignSlice(9, 0, contact_b_rel_cm_W.crossProduct(contact_normal_W));

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
