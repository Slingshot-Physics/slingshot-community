#ifndef INTEGRATOR_HEADER
#define INTEGRATOR_HEADER

#define QUATERNION_REGULARIZATION 0.1

#include "slingshot_types.hpp"
#include "matrix33.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"

namespace oy
{

namespace integrator
{
   // Time of integration needs to come from somewhere else.
   static float dt_ = 1.0f/500.f;

   Vector3 angVelDot(
      const Vector3 & netTorque,
      const Vector3 & angVel,
      const Matrix33 & J
   );

   Quaternion attitudeDot(
      const Quaternion & q,
      const Vector3 & angVel
   );

   Vector3 linVelDot(
      const Vector3 & netForce,
      float mass
   );

   void forwardStep(
      oy::types::rigidBody_t & rb,
      const oy::types::generalizedForce_t & forque,
      float dt=dt_
   );

   void setTimestep(float dtNew);
}

}

#endif
