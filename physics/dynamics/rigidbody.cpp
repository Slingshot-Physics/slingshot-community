#include "rigidbody.hpp"

namespace oy
{

namespace rb
{

   // Apply forces in global or body frame. The frame determines how the force
   // and the position of the force are applied.
   // If the frame is 'global', then the position of force application is assumed
   // to be relative to the body's center of mass in world coordinates.
   // If the frame is 'body', then the position of force application is assumed
   // to be relative to the body's center of mass in body coordinates (which
   // should be at the origin).
   // If 'position' is close to the zero vector, then no torque is produced as a
   // result of the force being applied at a contact point.
   void applyForce(
      const oy::types::rigidBody_t & body,
      const Vector3 & force,
      const Vector3 & position,
      const oy::types::enumFrame_t frame,
      oy::types::generalizedForce_t & general_force
   )
   {
      switch(frame)
      {
         case oy::types::enumFrame_t::GLOBAL:
         {
            general_force.appliedForce += force;
            if (position.magnitude() >= 1e-7f)
            {
               applyTorque(
                  body,
                  position.crossProduct(force),
                  frame,
                  general_force
               );
            }
            break;
         }
         case oy::types::enumFrame_t::BODY:
         {
            general_force.appliedForce += body.ql2b.conjugateSandwich(force);
            if (position.magnitude() >= 1e-7f)
            {
               applyTorque(
                  body,
                  body.ql2b.conjugateSandwich(position.crossProduct(force)),
                  frame,
                  general_force
               );
            }
            break;
         }
      }
   }

   // Applies a force on a body at its center of mass in either body frame or
   // global frame.
   void applyForce(
      const oy::types::rigidBody_t & body,
      const Vector3 & force,
      const oy::types::enumFrame_t frame,
      oy::types::generalizedForce_t & general_force
   )
   {
      switch(frame)
      {
         case oy::types::enumFrame_t::GLOBAL:
         {
            general_force.appliedForce += force;
            break;
         }
         case oy::types::enumFrame_t::BODY:
         {
            general_force.appliedForce += body.ql2b.conjugateSandwich(force);
            break;
         }
      }
   }


   void applyForce(
      const Vector3 & force,
      const oy::types::enumFrame_t frame,
      const oy::types::rk4Midpoint_t & body,
      oy::types::generalizedForce_t & general_force
   )
   {
      switch(frame)
      {
         case oy::types::enumFrame_t::GLOBAL:
         {
            general_force.appliedForce += force;
            break;
         }
         case oy::types::enumFrame_t::BODY:
         {
            general_force.appliedForce += body.ql2b.conjugateSandwich(force);
            break;
         }
      }
   }

   void applyForce(
      const Vector3 & force,
      const Vector3 & position,
      const oy::types::enumFrame_t frame,
      const oy::types::rk4Midpoint_t & body,
      oy::types::generalizedForce_t & general_force
   )
   {
      switch(frame)
      {
         case oy::types::enumFrame_t::GLOBAL:
         {
            general_force.appliedForce += force;
            if (position.magnitude() >= 1e-7f)
            {
               applyTorque(
                  position.crossProduct(force),
                  frame,
                  body,
                  general_force
               );
            }
            break;
         }
         case oy::types::enumFrame_t::BODY:
         {
            general_force.appliedForce += body.ql2b.conjugateSandwich(force);
            if (position.magnitude() >= 1e-7f)
            {
               applyTorque(
                  body.ql2b.conjugateSandwich(position.crossProduct(force)),
                  frame,
                  body,
                  general_force
               );
            }
            break;
         }
      }
   }

   // Apply torques in a given frame.
   void applyTorque(
      const oy::types::rigidBody_t & body,
      const Vector3 & torque,
      const oy::types::enumFrame_t frame,
      oy::types::generalizedForce_t & general_force
   )
   {
      // The attitude quaternion translates from lab frame to body frame, which
      // requires torques and angular velocities measured in body frame.
      switch(frame)
      {
         case oy::types::enumFrame_t::GLOBAL:
         {
            general_force.appliedTorque += body.ql2b.sandwich(torque);
            break;
         }
         case oy::types::enumFrame_t::BODY:
         {
            general_force.appliedTorque += torque;
            break;
         }
         default:
            break;
      }
   }

   void applyTorque(
      const Vector3 & torque,
      const oy::types::enumFrame_t frame,
      const oy::types::rk4Midpoint_t & body,
      oy::types::generalizedForce_t & general_force
   )
   {
      switch(frame)
      {
         case oy::types::enumFrame_t::GLOBAL:
         {
            general_force.appliedTorque += body.ql2b.sandwich(torque);
            break;
         }
         case oy::types::enumFrame_t::BODY:
         {
            general_force.appliedTorque += torque;
            break;
         }
         default:
            break;
      }
   }

   // Linear impulses are in global frame by default.
   void applyLinearImpulse(
      oy::types::rigidBody_t & body,
      const Vector3 & impulse,
      const oy::types::enumFrame_t frame
   )
   {
      switch(frame)
      {
         case oy::types::enumFrame_t::GLOBAL:
         {
            body.linVel += impulse / body.mass;
            break;
         }
         case oy::types::enumFrame_t::BODY:
         {
            body.linVel += body.ql2b.conjugateSandwich(impulse / body.mass);
            break;
         }
         default:
            break;
      }
   }

   // Angular impulses are in body frame by default.
   void applyAngularImpulse(
      oy::types::rigidBody_t & body,
      const Vector3 & impulse,
      const oy::types::enumFrame_t frame
   )
   {
      switch(frame)
      {
         case oy::types::enumFrame_t::GLOBAL:
         {
            body.angVel += (~body.inertiaTensor) * (body.ql2b.sandwich(impulse));
            break;
         }
         case oy::types::enumFrame_t::BODY:
         {
            body.angVel += (~body.inertiaTensor) * impulse;
            break;
         }
         default:
            break;
      }
   }

}

}
