#ifndef SLINGSHOT_TYPE_CONVERTERS_HEADER
#define SLINGSHOT_TYPE_CONVERTERS_HEADER

#include "slingshot_types.hpp"
#include "matrix33.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"

namespace oy
{

namespace converters
{

   void to_pod(
      const Quaternion & data_in,
      data_quaternion_t * data_out
   );

   void from_pod(
      const data_quaternion_t * data_in,
      Quaternion & data_out
   );

   void to_pod(
      const Matrix33 & data_in,
      data_matrix33_t * data_out
   );

   void from_pod(
      const data_matrix33_t * data_in,
      Matrix33 & data_out
   );

   void to_pod(
      const Vector3 & data_in,
      data_vector3_t * data_out
   );

   void from_pod(
      const data_vector3_t * data_in,
      Vector3 & data_out
   );

   void to_pod(
      const oy::types::orientedMass_t & oriented_mass_in,
      const oy::types::generalizedVelocity_t & generalized_vel_in,
      data_rigidbody_t * data_out
   );

   void from_pod(
      const data_rigidbody_t * data_in,
      oy::types::orientedMass_t & oriented_mass_in,
      oy::types::generalizedVelocity_t & generalized_vel_in
   );

   void to_pod(
      const oy::types::rigidBody_t & data_in,
      data_rigidbody_t * data_out
   );

   void from_pod(
      const data_rigidbody_t * data_in,
      oy::types::rigidBody_t & data_out
   );

   void to_pod(
      const oy::types::isometricCollider_t & data_in,
      data_isometricCollider_t * data_out
   );

   void from_pod(
      const data_isometricCollider_t * data_in,
      oy::types::isometricCollider_t & data_out
   );

   void to_pod(
      const oy::types::multiCollider_t & data_in,
      data_multiCollider_t * data_out
   );

   void from_pod(
      const data_multiCollider_t * data_in,
      oy::types::multiCollider_t & data_out
   );

   void to_pod(
      const oy::types::enumFrame_t & data_in,
      data_frameType_t * data_out
   );

   void from_pod(
      const data_frameType_t * data_in,
      oy::types::enumFrame_t & data_out
   );

   void to_pod(
      const logger::types::logging::enum_t & data_in,
      data_loggingType_t * data_out
   );

   void from_pod(
      const data_loggingType_t * data_in,
      logger::types::logging::enum_t & data_out
   );

   void to_pod(
      const logger::types::loggerConfig_t & data_in,
      data_loggerConfig_t * data_out
   );

   void from_pod(
      const data_loggerConfig_t * data_in,
      logger::types::loggerConfig_t & data_out
   );

   void to_pod(
      const oy::types::constraintBalljoint_t & data_in,
      data_constraintBalljoint_t * data_out
   );

   void from_pod(
      const data_constraintBalljoint_t * data_in,
      oy::types::constraintBalljoint_t & data_out
   );

   void to_pod(
      const oy::types::constraintCollision_t & data_in,
      data_constraintCollision_t * data_out
   );

   void from_pod(
      const data_constraintCollision_t * data_in,
      oy::types::constraintCollision_t & data_out
   );

   void to_pod(
      const oy::types::constraintFriction_t & data_in,
      data_constraintFriction_t * data_out
   );

   void from_pod(
      const data_constraintFriction_t * data_in,
      oy::types::constraintFriction_t & data_out
   );

   void to_pod(
      const oy::types::constraintGear_t & data_in,
      data_constraintGear_t * data_out
   );

   void from_pod(
      const data_constraintGear_t * data_in,
      oy::types::constraintGear_t & data_out
   );

   void to_pod(
      const oy::types::constraintRevoluteJoint_t & data_in,
      data_constraintRevoluteJoint_t * data_out
   );

   void from_pod(
      const data_constraintRevoluteJoint_t * data_in,
      oy::types::constraintRevoluteJoint_t & data_out
   );

   void to_pod(
      const oy::types::constraintRevoluteMotor_t & data_in,
      data_constraintRevoluteMotor_t * data_out
   );

   void from_pod(
      const data_constraintRevoluteMotor_t * data_in,
      oy::types::constraintRevoluteMotor_t & data_out
   );

   void to_pod(
      const oy::types::constraintRotation1d_t & data_in,
      data_constraintRotation1d_t * data_out
   );

   void from_pod(
      const data_constraintRotation1d_t * data_in,
      oy::types::constraintRotation1d_t & data_out
   );

   void to_pod(
      const oy::types::constraintTranslation1d_t & data_in,
      data_constraintTranslation1d_t * data_out
   );

   void from_pod(
      const data_constraintTranslation1d_t * data_in,
      oy::types::constraintTranslation1d_t & data_out
   );

   void to_pod(
      const oy::types::forceConstant_t & data_in,
      data_forceConstant_t * data_out
   );

   void from_pod(
      const data_forceConstant_t * data_in,
      oy::types::forceConstant_t & data_out
   );

   void to_pod(
      const oy::types::forceSpring_t & data_in,
      data_forceSpring_t * data_out
   );

   void from_pod(
      const data_forceSpring_t * data_in,
      oy::types::forceSpring_t & data_out
   );

   void to_pod(
      const oy::types::forceVelocityDamper_t & data_in,
      data_forceVelocityDamper_t * data_out
   );

   void from_pod(
      const data_forceVelocityDamper_t * data_in,
      oy::types::forceVelocityDamper_t & data_out
   );

   void to_pod(
      const oy::types::forceDrag_t & data_in,
      data_forceDrag_t * data_out
   );

   void from_pod(
      const data_forceDrag_t * data_in,
      oy::types::forceDrag_t & data_out
   );

   void to_pod(
      const oy::types::torqueDrag_t & data_in,
      data_torqueDrag_t * data_out
   );

   void from_pod(
      const data_torqueDrag_t * data_in,
      oy::types::torqueDrag_t & data_out
   );

   void to_pod(
      const oy::types::scenario_t & data_in,
      data_scenario_t * data_out
   );

   void from_pod(
      const data_scenario_t * data_in,
      oy::types::scenario_t & data_out
   );

}

}

#endif
