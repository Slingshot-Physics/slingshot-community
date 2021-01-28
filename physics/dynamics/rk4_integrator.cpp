#include "slingshot_type_converters.hpp"
#include "rk4_integrator.hpp"

namespace oy
{

namespace integrator
{
   Vector3 linVelDot(
      const Vector3 & netForce,
      float mass
   )
   {
      return netForce/mass;
   }

   // Assumes torques and angular velocity are in body frame.
   Vector3 angVelDot(
      const Vector3 & netTorque,
      const Vector3 & angVel,
      const Matrix33 & J
   )
   {
      return (~J)*(netTorque - angVel.crossProduct(J*angVel));
   }

   // This is in body frame.
   Quaternion attitudeDot(
      const Quaternion & q,
      const Vector3 & angVel
   )
   {
      Quaternion w(0.0, angVel);
      return 0.5f * q * w;
   }

   void forwardStep(
      oy::types::rigidBody_t & rb,
      const oy::types::generalizedForce_t & forque,
      float dt
   )
   {
      /********************************************************************
      * K1
      ********************************************************************/
      oy::types::rigidBody_t tempRb(rb);

      Vector3 v_k1 = linVelDot(forque.appliedForce, tempRb.mass);

      Vector3 x_k1 = tempRb.linVel;

      Vector3 w_k1 = angVelDot(forque.appliedTorque, tempRb.angVel, tempRb.inertiaTensor);

      Quaternion q_k1 = attitudeDot(tempRb.ql2b, tempRb.angVel);

      /********************************************************************
      * K2
      ********************************************************************/
      tempRb.linVel = rb.linVel + v_k1*dt/2.f;
      tempRb.linPos = rb.linPos + x_k1*dt/2.f;
      tempRb.angVel = rb.angVel + w_k1*dt/2.f;
      tempRb.ql2b = rb.ql2b + q_k1*dt/2;

      Vector3 v_k2 = linVelDot(forque.appliedForce, tempRb.mass);

      Vector3 x_k2 = tempRb.linVel;

      Vector3 w_k2 = angVelDot(forque.appliedTorque, tempRb.angVel, tempRb.inertiaTensor);

      Quaternion q_k2 = attitudeDot(tempRb.ql2b, tempRb.angVel);

      /********************************************************************
      * K3
      ********************************************************************/
      tempRb.linVel = rb.linVel + v_k2*dt/2.f;
      tempRb.linPos = rb.linPos + x_k2*dt/2.f;
      tempRb.angVel = rb.angVel + w_k2*dt/2.f;
      tempRb.ql2b = rb.ql2b + q_k2*dt/2;

      Vector3 v_k3 = linVelDot(forque.appliedForce, tempRb.mass);

      Vector3 x_k3 = tempRb.linVel;

      Vector3 w_k3 = angVelDot(forque.appliedTorque, tempRb.angVel, tempRb.inertiaTensor);

      Quaternion q_k3 = attitudeDot(tempRb.ql2b, tempRb.angVel);

      /********************************************************************
      * K4
      ********************************************************************/
      tempRb.linVel = rb.linVel + v_k3*dt;
      tempRb.linPos = rb.linPos + x_k3*dt;
      tempRb.angVel = rb.angVel + w_k3*dt;
      tempRb.ql2b = rb.ql2b + q_k3*dt;

      Vector3 v_k4 = linVelDot(forque.appliedForce, tempRb.mass);

      Vector3 x_k4 = tempRb.linVel;

      Vector3 w_k4 = angVelDot(forque.appliedTorque, tempRb.angVel, tempRb.inertiaTensor);

      Quaternion q_k4 = attitudeDot(tempRb.ql2b, tempRb.angVel);

      /********************************************************************
      * Updates
      ********************************************************************/
      rb.linPos = rb.linPos + (dt/6)*(x_k1 + 2*x_k2 + 2*x_k3 + x_k4);
      rb.linVel = rb.linVel + (dt/6)*(v_k1 + 2*v_k2 + 2*v_k3 + v_k4);
      rb.ql2b = rb.ql2b + (dt/6)*(q_k1 + 2*q_k2 + 2*q_k3 + q_k4);
      rb.angVel = rb.angVel + (dt/6)*(w_k1 + 2*w_k2 + 2*w_k3 + w_k4);
      if (fabs(1.0 - rb.ql2b.magnitude()) > 1e-8)
      {
         rb.ql2b = rb.ql2b + (QUATERNION_REGULARIZATION)*(1.0 - rb.ql2b.magnitudeSquared())*rb.ql2b;
      }
   }

   void setTimestep(float dtNew)
   {
      dt_ = dtNew;
   }

}

}
