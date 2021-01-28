#ifndef RIGIDBODY_HEADER
#define RIGIDBODY_HEADER

#include "slingshot_types.hpp"
#include "matrix33.hpp"
#include "vector3.hpp"

namespace oy
{

namespace rb
{
   // Apply a force to a body at a given point on the object in a given frame.
   // Positions in world coordinates must be relative to the body's center of
   // mass.
   void applyForce(
      const oy::types::rigidBody_t & body,
      const Vector3 & force,
      const Vector3 & position,
      const oy::types::enumFrame_t frame,
      oy::types::generalizedForce_t & general_force
   );

   // Apply a force to a body at the body's center of mass in the given frame.
   void applyForce(
      const oy::types::rigidBody_t & body,
      const Vector3 & force,
      const oy::types::enumFrame_t frame,
      oy::types::generalizedForce_t & general_force
   );

   // Apply a force to a body at a given point on the object in a given frame.
   // Positions in world coordinates must be relative to the body's center of
   // mass. The body's state is an intermediate state in RK4 integration.
   // The 'general_force' is the collection of all forces and torques that have
   // been applied to the given body.
   void applyForce(
      const Vector3 & force,
      const oy::types::enumFrame_t frame,
      const oy::types::rk4Midpoint_t & body,
      oy::types::generalizedForce_t & general_force
   );

   // Apply a force to a body at the body's center of mass in the given frame,
   // where the body's state is an intermediate state in RK4 integration.
   // The 'general_force' is the collection of all forces and torques that have
   // been applied to the given body.
   void applyForce(
      const Vector3 & force,
      const Vector3 & position,
      const oy::types::enumFrame_t frame,
      const oy::types::rk4Midpoint_t & body,
      oy::types::generalizedForce_t & general_force
   );

   // Apply a torque to a body in the given frame.
   void applyTorque(
      const oy::types::rigidBody_t & body,
      const Vector3 & torque,
      const oy::types::enumFrame_t frame,
      oy::types::generalizedForce_t & general_force
   );

   // Apply a torque to a body in the given frame, where the body's state is an
   // intermediate state in RK4 integration.
   // The 'general_force' is the collection of all forces and torques that have
   // been applied to the given body.
   void applyTorque(
      const Vector3 & torque,
      const oy::types::enumFrame_t frame,
      const oy::types::rk4Midpoint_t & body,
      oy::types::generalizedForce_t & general_force
   );

   // Apply a linear impulse to a body in the given frame.
   void applyLinearImpulse(
      oy::types::rigidBody_t & body,
      const Vector3 & linear_impulse,
      const oy::types::enumFrame_t frame
   );

   // Apply an angular impulse to a body in the given frame.
   void applyAngularImpulse(
      oy::types::rigidBody_t & body,
      const Vector3 & angular_impulse,
      const oy::types::enumFrame_t frame
   );
}

}

#endif
