#include "slingshot_type_converters.hpp"

#include "geometry_type_converters.hpp"

namespace oy
{

namespace converters
{

   void to_pod(
      const Quaternion & data_in,
      data_quaternion_t * data_out
   )
   {
      data_out->q[0] = data_in[0];
      data_out->q[1] = data_in[1];
      data_out->q[2] = data_in[2];
      data_out->q[3] = data_in[3];
   }

   void from_pod(
      const data_quaternion_t * data_in,
      Quaternion & data_out
   )
   {
      data_out[0] = data_in->q[0];
      data_out[1] = data_in->q[1];
      data_out[2] = data_in->q[2];
      data_out[3] = data_in->q[3];
   }

   void to_pod(
      const Matrix33 & data_in,
      data_matrix33_t * data_out
   )
   {
      data_out->m[0][0] = data_in(0, 0);
      data_out->m[0][1] = data_in(0, 1);
      data_out->m[0][2] = data_in(0, 2);
      data_out->m[1][0] = data_in(1, 0);
      data_out->m[1][1] = data_in(1, 1);
      data_out->m[1][2] = data_in(1, 2);
      data_out->m[2][0] = data_in(2, 0);
      data_out->m[2][1] = data_in(2, 1);
      data_out->m[2][2] = data_in(2, 2);
   }

   void from_pod(
      const data_matrix33_t * data_in,
      Matrix33 & data_out
   )
   {
      data_out(0, 0) = data_in->m[0][0];
      data_out(0, 1) = data_in->m[0][1];
      data_out(0, 2) = data_in->m[0][2];
      data_out(1, 0) = data_in->m[1][0];
      data_out(1, 1) = data_in->m[1][1];
      data_out(1, 2) = data_in->m[1][2];
      data_out(2, 0) = data_in->m[2][0];
      data_out(2, 1) = data_in->m[2][1];
      data_out(2, 2) = data_in->m[2][2];
   }

   void to_pod(
      const Vector3 & data_in,
      data_vector3_t * data_out
   )
   {
      data_out->v[0] = data_in[0];
      data_out->v[1] = data_in[1];
      data_out->v[2] = data_in[2];
   }

   void from_pod(
      const data_vector3_t * data_in,
      Vector3 & data_out
   )
   {
      data_out[0] = data_in->v[0];
      data_out[1] = data_in->v[1];
      data_out[2] = data_in->v[2];
   }

   void to_pod(
      const oy::types::orientedMass_t & oriented_mass_in,
      const oy::types::generalizedVelocity_t & generalized_vel_in,
      data_rigidbody_t * data_out
   )
   {
      initialize_rigidbody(data_out);
      data_out->mass = oriented_mass_in.mass;
      data_out->stationary = false;
      to_pod(oriented_mass_in.inertiaTensor, &(data_out->J));
      to_pod(oriented_mass_in.linPos, &(data_out->linpos));
      to_pod(oriented_mass_in.ql2b, &(data_out->orientation));

      to_pod(generalized_vel_in.linVel, &(data_out->linvel));
      to_pod(generalized_vel_in.angVel, &(data_out->rotvel));
   }

   void from_pod(
      const data_rigidbody_t * data_in,
      oy::types::orientedMass_t & oriented_mass_out,
      oy::types::generalizedVelocity_t & generalized_vel_out
   )
   {
      oriented_mass_out.mass = data_in->mass;
      from_pod(&(data_in->J), oriented_mass_out.inertiaTensor);
      from_pod(&(data_in->linpos), oriented_mass_out.linPos);
      from_pod(&(data_in->orientation), oriented_mass_out.ql2b);

      from_pod(&(data_in->linvel), generalized_vel_out.linVel);
      from_pod(&(data_in->rotvel), generalized_vel_out.angVel);
   }

   void to_pod(
      const oy::types::rigidBody_t & data_in,
      data_rigidbody_t * data_out
   )
   {
      initialize_rigidbody(data_out);
      data_out->mass = data_in.mass;
      to_pod(data_in.angVel, &data_out->rotvel);
      to_pod(data_in.linPos, &data_out->linpos);
      to_pod(data_in.linVel, &data_out->linvel);
      to_pod(data_in.ql2b, &data_out->orientation);
      to_pod(data_in.inertiaTensor, &data_out->J);
   }

   void from_pod(
      const data_rigidbody_t * data_in,
      oy::types::rigidBody_t & data_out
   )
   {
      data_out.mass = data_in->mass;
      from_pod(&(data_in->rotvel), data_out.angVel);
      from_pod(&(data_in->linpos), data_out.linPos);
      from_pod(&(data_in->linvel), data_out.linVel);
      from_pod(&(data_in->orientation), data_out.ql2b);
      from_pod(&(data_in->J), data_out.inertiaTensor);
   }

   void to_pod(
      const oy::types::isometricCollider_t & data_in,
      data_isometricCollider_t * data_out
   )
   {
      data_out->mu = data_in.mu;
      data_out->restitution = data_in.restitution;
      data_out->enabled = data_in.enabled;
   }

   void from_pod(
      const data_isometricCollider_t * data_in,
      oy::types::isometricCollider_t & data_out
   )
   {
      data_out.mu = data_in->mu;
      data_out.restitution = data_in->restitution;
      data_out.enabled = data_in->enabled;
   }

   void to_pod(
      const oy::types::multiCollider_t & data_in,
      data_multiCollider_t * data_out
   )
   {
      data_out->numColliders = data_in.numColliders;
      
      for (unsigned int i = 0; i < data_out->numColliders; ++i)
      {
         to_pod(data_in.colliders[i], &(data_out->colliders[i]));
      }
   }

   void from_pod(
      const data_multiCollider_t * data_in,
      oy::types::multiCollider_t & data_out
   )
   {
      data_out.numColliders = data_in->numColliders;
      
      for (unsigned int i = 0; i < data_out.numColliders; ++i)
      {
         from_pod(&(data_in->colliders[i]), data_out.colliders[i]);
      }
   }

   void to_pod(
      const oy::types::enumFrame_t & data_in,
      data_frameType_t * data_out
   )
   {
      int temp = static_cast<int>(data_in);
      *data_out = (data_frameType_t)temp;
   }

   void from_pod(
      const data_frameType_t * data_in,
      oy::types::enumFrame_t & data_out
   )
   {
      int temp = static_cast<int>(*data_in);
      data_out = (oy::types::enumFrame_t )temp;
   }

   void to_pod(
      const logger::types::logging::enum_t & data_in,
      data_loggingType_t * data_out
   )
   {
      int temp = data_in;
      *data_out = (data_loggingType_t )temp;
   }

   void from_pod(
      const data_loggingType_t * data_in,
      logger::types::logging::enum_t & data_out
   )
   {
      int temp = *data_in;
      data_out = (logger::types::logging::enum_t )temp;
   }

   void to_pod(
      const logger::types::loggerConfig_t & data_in,
      data_loggerConfig_t * data_out
   )
   {
      if (data_in.logDir.size() > 120)
      {
         strncpy(data_out->logDir, ".\0", 1);
      }
      else
      {
         strncpy(data_out->logDir, data_in.logDir.c_str(), data_in.logDir.size());
      }
      to_pod(data_in.logType, &(data_out->loggingType));
   }

   void from_pod(
      const data_loggerConfig_t * data_in,
      logger::types::loggerConfig_t & data_out
   )
   {
      data_out.logDir = std::string(data_in->logDir);
      from_pod(&(data_in->loggingType), data_out.logType);
   }

   void to_pod(
      const oy::types::constraintBalljoint_t & data_in,
      data_constraintBalljoint_t * data_out
   )
   {
      to_pod(data_in.parentLinkPoint, &(data_out->parentLinkPoint));
      to_pod(data_in.childLinkPoint, &(data_out->childLinkPoint));
   }

   void from_pod(
      const data_constraintBalljoint_t * data_in,
      oy::types::constraintBalljoint_t & data_out
   )
   {
      from_pod(&(data_in->parentLinkPoint), data_out.parentLinkPoint);
      from_pod(&(data_in->childLinkPoint), data_out.childLinkPoint);
   }

   void to_pod(
      const oy::types::constraintCollision_t & data_in,
      data_constraintCollision_t * data_out
   )
   {
      data_out->restitution = data_in.restitution;
      to_pod(data_in.bodyAContact, &(data_out->bodyAContact));
      to_pod(data_in.bodyBContact, &(data_out->bodyBContact));
      to_pod(data_in.unitNormal, &(data_out->n_hat));
   }

   void from_pod(
      const data_constraintCollision_t * data_in,
      oy::types::constraintCollision_t & data_out
   )
   {
      data_out.restitution = data_in->restitution;
      from_pod(&(data_in->bodyAContact), data_out.bodyAContact);
      from_pod(&(data_in->bodyBContact), data_out.bodyBContact);
      from_pod(&(data_in->n_hat), data_out.unitNormal);
   }

   void to_pod(
      const oy::types::constraintFriction_t & data_in,
      data_constraintFriction_t * data_out
   )
   {
      data_out->muTotal = data_in.muTotal;
      to_pod(data_in.bodyAContact, &(data_out->bodyAContact));
      to_pod(data_in.bodyBContact, &(data_out->bodyBContact));
      to_pod(data_in.unitNormal, &(data_out->unitNormal));
   }

   void from_pod(
      const data_constraintFriction_t * data_in,
      oy::types::constraintFriction_t & data_out
   )
   {
      data_out.muTotal = data_in->muTotal;
      from_pod(&(data_in->bodyAContact), data_out.bodyAContact);
      from_pod(&(data_in->bodyBContact), data_out.bodyBContact);
      from_pod(&(data_in->unitNormal), data_out.unitNormal);
   }

   void to_pod(
      const oy::types::constraintGear_t & data_in,
      data_constraintGear_t * data_out
   )
   {
      data_out->parentGearRadius = data_in.parentGearRadius;
      data_out->childGearRadius = data_in.childGearRadius;

      to_pod(data_in.parentAxis, &(data_out->parentAxis));
      to_pod(data_in.childAxis, &(data_out->childAxis));

      data_out->rotateParallel = (unsigned int )data_in.rotateParallel;
   }

   void from_pod(
      const data_constraintGear_t * data_in,
      oy::types::constraintGear_t & data_out
   )
   {
      data_out.parentGearRadius = data_in->parentGearRadius;
      data_out.childGearRadius = data_in->childGearRadius;

      from_pod(&(data_in->parentAxis), data_out.parentAxis);
      from_pod(&(data_in->childAxis), data_out.childAxis);

      data_out.rotateParallel = (bool )data_in->rotateParallel;
   }

   void to_pod(
      const oy::types::constraintRevoluteJoint_t & data_in,
      data_constraintRevoluteJoint_t * data_out
   )
   {
      to_pod(data_in.parentLinkPoints[0], &(data_out->parentLinkPoints[0]));
      to_pod(data_in.parentLinkPoints[1], &(data_out->parentLinkPoints[1]));
      to_pod(data_in.childLinkPoints[0], &(data_out->childLinkPoints[0]));
      to_pod(data_in.childLinkPoints[1], &(data_out->childLinkPoints[1]));
   }

   void from_pod(
      const data_constraintRevoluteJoint_t * data_in,
      oy::types::constraintRevoluteJoint_t & data_out
   )
   {
      from_pod(&data_in->parentLinkPoints[0], data_out.parentLinkPoints[0]);
      from_pod(&data_in->parentLinkPoints[1], data_out.parentLinkPoints[1]);
      from_pod(&data_in->childLinkPoints[0], data_out.childLinkPoints[0]);
      from_pod(&data_in->childLinkPoints[1], data_out.childLinkPoints[1]);
   }

   void to_pod(
      const oy::types::constraintRevoluteMotor_t & data_in,
      data_constraintRevoluteMotor_t * data_out
   )
   {
      to_pod(data_in.parentAxis, &(data_out->parentAxis));
      to_pod(data_in.childAxis, &(data_out->childAxis));
      data_out->maxTorque = data_in.maxTorque;
      data_out->angularSpeed = data_in.angularSpeed;
   }

   void from_pod(
      const data_constraintRevoluteMotor_t * data_in,
      oy::types::constraintRevoluteMotor_t & data_out
   )
   {
      from_pod(&(data_in->parentAxis), data_out.parentAxis);
      from_pod(&(data_in->childAxis), data_out.childAxis);
      data_out.maxTorque = data_in->maxTorque;
      data_out.angularSpeed = data_in->angularSpeed;
   }

   void to_pod(
      const oy::types::constraintRotation1d_t & data_in,
      data_constraintRotation1d_t * data_out
   )
   {
      to_pod(data_in.parentAxis, &(data_out->parentAxis));
      to_pod(data_in.childAxis, &(data_out->childAxis));
   }

   void from_pod(
      const data_constraintRotation1d_t * data_in,
      oy::types::constraintRotation1d_t & data_out
   )
   {
      from_pod(&(data_in->parentAxis), data_out.parentAxis);
      from_pod(&(data_in->childAxis), data_out.childAxis);
   }

   void to_pod(
      const oy::types::constraintTranslation1d_t & data_in,
      data_constraintTranslation1d_t * data_out
   )
   {
      to_pod(data_in.parentAxis, &(data_out->parentAxis));
      to_pod(data_in.parentLinkPoint, &(data_out->parentLinkPoint));
      to_pod(data_in.childLinkPoint, &(data_out->childLinkPoint));
   }

   void from_pod(
      const data_constraintTranslation1d_t * data_in,
      oy::types::constraintTranslation1d_t & data_out
   )
   {
      from_pod(&data_in->parentAxis, data_out.parentAxis);
      from_pod(&data_in->parentLinkPoint, data_out.parentLinkPoint);
      from_pod(&data_in->childLinkPoint, data_out.childLinkPoint);
   }

   void to_pod(
      const oy::types::forceConstant_t & data_in,
      data_forceConstant_t * data_out
   )
   {
      to_pod(data_in.childLinkPoint, &(data_out->childLinkPoint));
      to_pod(data_in.acceleration, &(data_out->acceleration));
      to_pod(data_in.forceFrame, &(data_out->frame));
   }

   void from_pod(
      const data_forceConstant_t * data_in,
      oy::types::forceConstant_t & data_out
   )
   {
      from_pod(&(data_in->childLinkPoint), data_out.childLinkPoint);
      from_pod(&(data_in->acceleration), data_out.acceleration);
      from_pod(&(data_in->frame), data_out.forceFrame);
   }

   void to_pod(
      const oy::types::forceSpring_t & data_in,
      data_forceSpring_t * data_out
   )
   {
      data_out->restLength = data_in.restLength;
      data_out->springCoeff = data_in.springCoeff;
      to_pod(data_in.parentLinkPoint, &(data_out->parentLinkPoint));
      to_pod(data_in.childLinkPoint, &(data_out->childLinkPoint));
   }

   void from_pod(
      const data_forceSpring_t * data_in,
      oy::types::forceSpring_t & data_out
   )
   {
      data_out.restLength = data_in->restLength;
      data_out.springCoeff = data_in->springCoeff;
      from_pod(&(data_in->parentLinkPoint), data_out.parentLinkPoint);
      from_pod(&(data_in->childLinkPoint), data_out.childLinkPoint);
   }

   void to_pod(
      const oy::types::forceVelocityDamper_t & data_in,
      data_forceVelocityDamper_t * data_out
   )
   {
      data_out->damperCoeff = data_in.damperCoeff;
      to_pod(data_in.parentLinkPoint, &(data_out->parentLinkPoint));
      to_pod(data_in.childLinkPoint, &(data_out->childLinkPoint));
   }

   void from_pod(
      const data_forceVelocityDamper_t * data_in,
      oy::types::forceVelocityDamper_t & data_out
   )
   {
      data_out.damperCoeff = data_in->damperCoeff;
      from_pod(&(data_in->parentLinkPoint), data_out.parentLinkPoint);
      from_pod(&(data_in->childLinkPoint), data_out.childLinkPoint);
   }

   void to_pod(
      const oy::types::forceDrag_t & data_in,
      data_forceDrag_t * data_out
   )
   {
      data_out->linearDragCoeff = data_in.linearDragCoeff;
      data_out->quadraticDragCoeff = data_in.quadraticDragCoeff;
   }

   void from_pod(
      const data_forceDrag_t * data_in,
      oy::types::forceDrag_t & data_out
   )
   {
      data_out.linearDragCoeff = data_in->linearDragCoeff;
      data_out.quadraticDragCoeff = data_in->quadraticDragCoeff;
   }

   void to_pod(
      const oy::types::torqueDrag_t & data_in,
      data_torqueDrag_t * data_out
   )
   {
      data_out->linearDragCoeff = data_in.linearDragCoeff;
      data_out->quadraticDragCoeff = data_in.quadraticDragCoeff;
   }

   void from_pod(
      const data_torqueDrag_t * data_in,
      oy::types::torqueDrag_t & data_out
   )
   {
      data_out.linearDragCoeff = data_in->linearDragCoeff;
      data_out.quadraticDragCoeff = data_in->quadraticDragCoeff;
   }

   void to_pod(
      const oy::types::scenario_t & data_in,
      data_scenario_t * data_out
   )
   {
      // Free any existing memory in the scenario.
      // I will absolutely nuke all of your data if you let me.
      initialize_scenario(data_out);

      int i = 0;

      data_out->numBalljointConstraints = data_in.balljoints.size();
      data_out->numGearConstraints = data_in.gears.size();
      data_out->numRevoluteJointConstraints = data_in.revolute_joints.size();
      data_out->numRevoluteMotorConstraints = data_in.revolute_motors.size();
      data_out->numRotation1dConstraints = data_in.rotation_1d.size();
      data_out->numTranslation1dConstraints = data_in.translation_1d.size();
      data_out->numConstantForces = data_in.constant_forces.size();
      data_out->numSpringForces = data_in.springs.size();
      data_out->numVelocityDamperForces = data_in.dampers.size();
      data_out->numDragTorques = data_in.drag_torques.size();
      data_out->numDragForces = data_in.drag_forces.size();
      data_out->numShapes = data_in.shapes.size();
      data_out->numBodies = data_in.bodies.size();
      data_out->numIsometricColliders = data_in.isometric_colliders.size();

      data_out->balljointConstraints = (data_constraintBalljoint_t *)malloc(
         sizeof(data_constraintBalljoint_t) * data_out->numBalljointConstraints
      );
      i = 0;
      for (
         auto bj_it = data_in.balljoints.begin();
         bj_it != data_in.balljoints.end();
         ++bj_it, ++i
      )
      {
         data_constraintBalljoint_t tempBalljoint;
         initialize_constraintBalljoint(&tempBalljoint);
         to_pod(bj_it->second, &tempBalljoint);
         tempBalljoint.parentId = bj_it->first.parentId;
         tempBalljoint.childId = bj_it->first.childId;
         data_out->balljointConstraints[i] = tempBalljoint;
      }

      data_out->gearConstraints = (data_constraintGear_t *)malloc(
         sizeof(data_constraintGear_t) * data_out->numGearConstraints
      );
      i = 0;
      for (
         auto ge_it = data_in.gears.begin();
         ge_it != data_in.gears.end();
         ++ge_it, ++i
      )
      {
         data_constraintGear_t tempGear;
         initialize_constraintGear(&tempGear);
         to_pod(ge_it->second, &tempGear);
         tempGear.parentId = ge_it->first.parentId;
         tempGear.childId = ge_it->first.childId;
         data_out->gearConstraints[i] = tempGear;
      }

      data_out->revoluteJointConstraints = (data_constraintRevoluteJoint_t *)malloc(
         sizeof(data_constraintRevoluteJoint_t) * data_out->numRevoluteJointConstraints
      );
      i = 0;
      for (
         auto rj_it = data_in.revolute_joints.begin();
         rj_it != data_in.revolute_joints.end();
         ++rj_it, ++i
      )
      {
         data_constraintRevoluteJoint_t tempRevoluteJoint;
         initialize_constraintRevoluteJoint(&tempRevoluteJoint);
         to_pod(rj_it->second, &tempRevoluteJoint);
         tempRevoluteJoint.parentId = rj_it->first.parentId;
         tempRevoluteJoint.childId = rj_it->first.childId;
         data_out->revoluteJointConstraints[i] = tempRevoluteJoint;
      }

      data_out->revoluteMotorConstraints = (data_constraintRevoluteMotor_t *)malloc(
         sizeof(data_constraintRevoluteMotor_t) * data_out->numRevoluteMotorConstraints
      );
      i = 0;
      for (
         auto rm_it = data_in.revolute_motors.begin();
         rm_it != data_in.revolute_motors.end();
         ++rm_it, ++i
      )
      {
         data_constraintRevoluteMotor_t tempRevoluteMotor;
         initialize_constraintRevoluteMotor(&tempRevoluteMotor);
         to_pod(rm_it->second, &tempRevoluteMotor);
         tempRevoluteMotor.parentId = rm_it->first.parentId;
         tempRevoluteMotor.childId = rm_it->first.childId;
         data_out->revoluteMotorConstraints[i] = tempRevoluteMotor;
      }

      data_out->rotation1dConstraints = (data_constraintRotation1d_t *)malloc(
         sizeof(data_constraintRotation1d_t) * data_out->numRotation1dConstraints
      );
      i = 0;
      for (
         auto ro_it = data_in.rotation_1d.begin();
         ro_it != data_in.rotation_1d.end();
         ++ro_it, ++i
      )
      {
         data_constraintRotation1d_t tempRotation;
         initialize_constraintRotation1d(&tempRotation);
         to_pod(ro_it->second, &tempRotation);
         tempRotation.parentId = ro_it->first.parentId;
         tempRotation.childId = ro_it->first.childId;
         data_out->rotation1dConstraints[i] = tempRotation;
      }

      data_out->translation1dConstraints = (data_constraintTranslation1d_t *)malloc(
         sizeof(data_constraintTranslation1d_t) * data_out->numTranslation1dConstraints
      );
      i = 0;
      for (
         auto tr_it = data_in.translation_1d.begin();
         tr_it != data_in.translation_1d.end();
         ++tr_it, ++i
      )
      {
         data_constraintTranslation1d_t tempTranslation;
         initialize_constraintTranslation1d(&tempTranslation);
         to_pod(tr_it->second, &tempTranslation);
         tempTranslation.parentId = tr_it->first.parentId;
         tempTranslation.childId = tr_it->first.childId;
         data_out->translation1dConstraints[i] = tempTranslation;
      }

      data_out->constantForces = (data_forceConstant_t *)malloc(
         sizeof(data_forceConstant_t) * data_out->numConstantForces
      );
      i = 0;
      for (
         auto cf_it = data_in.constant_forces.begin();
         cf_it != data_in.constant_forces.end();
         ++cf_it, ++i
      )
      {
         data_forceConstant_t tempConstantForce;
         initialize_forceConstant(&tempConstantForce);
         to_pod(cf_it->second, &tempConstantForce);
         tempConstantForce.childId = cf_it->first.childId;
         data_out->constantForces[i] = tempConstantForce;
      }

      data_out->springForces = (data_forceSpring_t *)malloc(
         sizeof(data_forceSpring_t) * data_out->numSpringForces
      );
      i = 0;
      for (
         auto sp_it = data_in.springs.begin();
         sp_it != data_in.springs.end();
         ++sp_it, ++i
      )
      {
         data_forceSpring_t tempSpring;
         initialize_forceSpring(&tempSpring);
         to_pod(sp_it->second, &tempSpring);
         tempSpring.parentId = sp_it->first.parentId;
         tempSpring.childId = sp_it->first.childId;
         data_out->springForces[i] = tempSpring;
      }

      data_out->velocityDamperForces = (data_forceVelocityDamper_t *)malloc(
         sizeof(data_forceVelocityDamper_t) * data_out->numVelocityDamperForces
      );
      i = 0;
      for (
         auto da_it = data_in.dampers.begin();
         da_it != data_in.dampers.end();
         ++da_it, ++i
      )
      {
         data_forceVelocityDamper_t tempDamper;
         initialize_forceVelocityDamper(&tempDamper);
         to_pod(da_it->second, &tempDamper);
         tempDamper.parentId = da_it->first.parentId;
         tempDamper.childId = da_it->first.childId;
         data_out->velocityDamperForces[i] = tempDamper;
      }

      data_out->dragForces = (data_forceDrag_t *)malloc(
         sizeof(data_forceDrag_t) * data_out->numDragForces
      );
      i = 0;
      for (
         auto df_it = data_in.drag_forces.begin();
         df_it != data_in.drag_forces.end();
         ++df_it, ++i
      )
      {
         data_forceDrag_t tempDrag;
         initialize_forceDrag(&tempDrag);
         to_pod(df_it->second, &tempDrag);
         tempDrag.childId = df_it->first.childId;
         data_out->dragForces[i] = tempDrag;
      }

      data_out->dragTorques = (data_torqueDrag_t *)malloc(
         sizeof(data_torqueDrag_t) * data_out->numDragTorques
      );
      i = 0;
      for (
         auto dr_it = data_in.drag_torques.begin();
         dr_it != data_in.drag_torques.end();
         ++dr_it, ++i
      )
      {
         data_torqueDrag_t tempDrag;
         initialize_torqueDrag(&tempDrag);
         to_pod(dr_it->second, &tempDrag);
         tempDrag.childId = dr_it->first.childId;
         data_out->dragTorques[i] = tempDrag;
      }

      data_out->bodies = (data_rigidbody_t *)malloc(
         sizeof(data_rigidbody_t) * data_out->numBodies
      );
      i = 0;
      for (
         auto body_it = data_in.bodies.begin();
         body_it != data_in.bodies.end();
         ++body_it, ++i
      )
      {
         data_rigidbody_t tempBody;
         initialize_rigidbody(&tempBody);
         to_pod(body_it->second, &tempBody);
         tempBody.id = body_it->first;
         tempBody.stationary = (data_in.body_types.at(body_it->first) == oy::types::enumRigidBody_t::STATIONARY);
         data_out->bodies[i] = tempBody;
      }

      data_out->isometricColliders = (data_isometricCollider_t *)malloc(
         sizeof(data_isometricCollider_t) * data_out->numIsometricColliders
      );
      i = 0;
      for (
         auto coll_it = data_in.isometric_colliders.begin();
         coll_it != data_in.isometric_colliders.end();
         ++coll_it, ++i
      )
      {
         data_isometricCollider_t tempIsometricCollider;
         initialize_isometricCollider(&tempIsometricCollider);
         to_pod(coll_it->second, &tempIsometricCollider);
         tempIsometricCollider.bodyId = coll_it->first;
         data_out->isometricColliders[i] = tempIsometricCollider;
      }

      data_out->shapes = (data_shape_t *)malloc(
         sizeof(data_shape_t) * data_out->numShapes
      );
      i = 0;
      for (
         auto shape_it = data_in.shapes.begin();
         shape_it != data_in.shapes.end();
         ++shape_it, ++i
      )
      {
         data_shape_t tempShape;
         initialize_shape(&tempShape);
         geometry::converters::to_pod(
            shape_it->second, &tempShape
         );
         tempShape.bodyId = shape_it->first;
         data_out->shapes[i] = tempShape;
      }
   }

   void from_pod(
      const data_scenario_t * data_in,
      oy::types::scenario_t & data_out
   )
   {
      for (int i = 0; i < data_in->numBodies; ++i)
      {
         oy::types::rigidBody_t tempBody;
         from_pod(&(data_in->bodies[i]), tempBody);
         data_out.bodies[data_in->bodies[i].id] = tempBody;
         data_out.body_types[data_in->bodies[i].id] = (
            data_in->bodies[i].stationary ? oy::types::enumRigidBody_t::STATIONARY : oy::types::enumRigidBody_t::DYNAMIC
         );
      }

      for (int i = 0; i < data_in->numIsometricColliders; ++i)
      {
         oy::types::isometricCollider_t tempIsometricCollider;
         from_pod(&(data_in->isometricColliders[i]), tempIsometricCollider);
         data_out.isometric_colliders[data_in->isometricColliders[i].bodyId] = tempIsometricCollider;
      }

      for (int i = 0; i < data_in->numShapes; ++i)
      {
         geometry::types::shape_t tempShape;
         geometry::converters::from_pod(&(data_in->shapes[i]), tempShape);
         data_out.shapes[data_in->shapes[i].bodyId] = tempShape;
      }

      for (int i = 0; i < data_in->numBalljointConstraints; ++i)
      {
         oy::types::constraintBalljoint_t tempBalljoint;
         from_pod(&(data_in->balljointConstraints[i]), tempBalljoint);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = data_in->balljointConstraints[i].parentId;
         bodyLink.childId = data_in->balljointConstraints[i].childId;
         data_out.balljoints.push_back({bodyLink, tempBalljoint});
      }

      for (int i = 0; i < data_in->numGearConstraints; ++i)
      {
         oy::types::constraintGear_t tempGear;
         from_pod(&(data_in->gearConstraints[i]), tempGear);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = data_in->gearConstraints[i].parentId;
         bodyLink.childId = data_in->gearConstraints[i].childId;
         data_out.gears.push_back({bodyLink, tempGear});
      }

      for (int i = 0; i < data_in->numRevoluteJointConstraints; ++i)
      {
         oy::types::constraintRevoluteJoint_t tempRevoluteJoint;
         from_pod(&(data_in->revoluteJointConstraints[i]), tempRevoluteJoint);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = data_in->revoluteJointConstraints[i].parentId;
         bodyLink.childId = data_in->revoluteJointConstraints[i].childId;
         data_out.revolute_joints.push_back({bodyLink, tempRevoluteJoint});
      }

      for (int i = 0; i < data_in->numRevoluteMotorConstraints; ++i)
      {
         oy::types::constraintRevoluteMotor_t tempRevoluteMotor;
         from_pod(&(data_in->revoluteMotorConstraints[i]), tempRevoluteMotor);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = data_in->revoluteMotorConstraints[i].parentId;
         bodyLink.childId = data_in->revoluteMotorConstraints[i].childId;
         data_out.revolute_motors.push_back({bodyLink, tempRevoluteMotor});
      }

      for (int i = 0; i < data_in->numRotation1dConstraints; ++i)
      {
         oy::types::constraintRotation1d_t tempRotation;
         from_pod(&(data_in->rotation1dConstraints[i]), tempRotation);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = data_in->rotation1dConstraints[i].parentId;
         bodyLink.childId = data_in->rotation1dConstraints[i].childId;
         data_out.rotation_1d.push_back({bodyLink, tempRotation});
      }

      for (int i = 0; i < data_in->numTranslation1dConstraints; ++i)
      {
         oy::types::constraintTranslation1d_t tempTranslation;
         from_pod(&(data_in->translation1dConstraints[i]), tempTranslation);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = data_in->translation1dConstraints[i].parentId;
         bodyLink.childId = data_in->translation1dConstraints[i].childId;
         data_out.translation_1d.push_back({bodyLink, tempTranslation});
      }

      for (int i = 0; i < data_in->numConstantForces; ++i)
      {
         oy::types::forceConstant_t tempConstantForce;
         from_pod(&(data_in->constantForces[i]), tempConstantForce);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = -1;
         bodyLink.childId = data_in->constantForces[i].childId;
         data_out.constant_forces.push_back({bodyLink, tempConstantForce});
      }

      for (int i = 0; i < data_in->numSpringForces; ++i)
      {
         oy::types::forceSpring_t tempSpring;
         from_pod(&(data_in->springForces[i]), tempSpring);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = data_in->springForces[i].parentId;
         bodyLink.childId = data_in->springForces[i].childId;
         data_out.springs.push_back({bodyLink, tempSpring});
      }

      for (int i = 0; i < data_in->numVelocityDamperForces; ++i)
      {
         oy::types::forceVelocityDamper_t tempDamper;
         from_pod(&(data_in->velocityDamperForces[i]), tempDamper);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = data_in->velocityDamperForces[i].parentId;
         bodyLink.childId = data_in->velocityDamperForces[i].childId;
         data_out.dampers.push_back({bodyLink, tempDamper});
      }

      for (int i = 0; i < data_in->numDragTorques; ++i)
      {
         oy::types::torqueDrag_t tempDrag;
         from_pod(&(data_in->dragTorques[i]), tempDrag);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = -1;
         bodyLink.childId = data_in->dragTorques[i].childId;
         data_out.drag_torques.push_back({bodyLink, tempDrag});
      }

      for (int i = 0; i < data_in->numDragForces; ++i)
      {
         oy::types::forceDrag_t tempDrag;
         from_pod(&(data_in->dragForces[i]), tempDrag);
         oy::types::bodyLink_t bodyLink;
         bodyLink.parentId = -1;
         bodyLink.childId = data_in->dragForces[i].childId;
         data_out.drag_forces.push_back({bodyLink, tempDrag});
      }

      from_pod(&(data_in->logger), data_out.logger);
   }

}

}
