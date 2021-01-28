#include "handle.hpp"

#include "data_model_io.h"

#include "slingshot_type_converters.hpp"
#include "geometry_type_converters.hpp"
#include "logger_utils.hpp"
#include "raycast.hpp"
#include "rk4_integrator.hpp"
#include "static_logger.hpp"
#include "transform.hpp"

#include <iostream>

namespace oy
{
   Handle::Handle(void)
      : allocator_(nullptr)
      , rigid_body_query_(0)
      , balljoint_constraint_query_(0)
      , gear_constraint_query_(0)
      , revolute_joint_constraint_query_(0)
      , revolute_motor_constraint_query_(0)
      , rotation_1d_constraint_query_(0)
      , translation_1d_constraint_query_(0)
      , constant_force_query_(0)
      , drag_force_query_(0)
      , spring_force_query_(0)
      , velocity_damper_force_query_(0)
      , drag_torque_query_(0)
      , frame_counter_(0)
   {
      allocator_ = new trecs::Allocator(20000);

      registerSystems();
      allocator_->initializeSystems();
      registerQueriesAndComponents();
   }

   Handle::Handle(const data_scenario_t & scenario)
      : allocator_(nullptr)
      , rigid_body_query_(0)
      , balljoint_constraint_query_(0)
      , gear_constraint_query_(0)
      , revolute_joint_constraint_query_(0)
      , revolute_motor_constraint_query_(0)
      , rotation_1d_constraint_query_(0)
      , translation_1d_constraint_query_(0)
      , constant_force_query_(0)
      , drag_force_query_(0)
      , spring_force_query_(0)
      , velocity_damper_force_query_(0)
      , drag_torque_query_(0)
      , frame_counter_(0)
   {
      allocator_ = new trecs::Allocator(20000);

      registerSystems();
      allocator_->initializeSystems();
      registerQueriesAndComponents();

      oy::converters::from_pod(&scenario, scenario_);
      setupScenario(scenario_);
   }

   Handle::~Handle(void)
   {
      if (allocator_ != nullptr)
      {
         delete allocator_;
         allocator_ = nullptr;
      }
   }

   void Handle::registerSystems(void)
   {
      rk4System_ = allocator_->registerSystem<oy::Rk4Integrator>();
      forque_resetter_system_ = allocator_->registerSystem<oy::GeneralizedForceResetSystem>();
      rk4_finalizer_system_ = allocator_->registerSystem<oy::Rk4Finalizer>();
      rk4_midpoint0_system_ = allocator_->registerSystem<oy::Rk4MidpointCalculator<0> >();
      rk4_midpoint1_system_ = allocator_->registerSystem<oy::Rk4MidpointCalculator<1> >();
      rk4_midpoint2_system_ = allocator_->registerSystem<oy::Rk4MidpointCalculator<2> >();
      rk4_midpoint3_system_ = allocator_->registerSystem<oy::Rk4MidpointCalculator<3> >();
      rk4_increment0_system_ = allocator_->registerSystem<oy::Rk4IncrementCalculator<0> >();
      rk4_increment1_system_ = allocator_->registerSystem<oy::Rk4IncrementCalculator<1> >();
      rk4_increment2_system_ = allocator_->registerSystem<oy::Rk4IncrementCalculator<2> >();
      rk4_increment3_system_ = allocator_->registerSystem<oy::Rk4IncrementCalculator<3> >();

      cube_aabb_system_ = allocator_->registerSystem<oy::AabbCalculator<cube_t> >();
      sphere_aabb_system_ = allocator_->registerSystem<oy::AabbCalculator<sphere_t> >();
      capsule_aabb_system_ = allocator_->registerSystem<oy::AabbCalculator<capsule_t> >();
      cylinder_aabb_system_ = allocator_->registerSystem<oy::AabbCalculator<cylinder_t> >();

/// broadphase
      cube_cube_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<cube_t, cube_t> >();
      cube_sphere_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<cube_t, sphere_t> >();
      cube_capsule_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<cube_t, capsule_t> >();
      cube_cylinder_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<cube_t, cylinder_t> >();
      sphere_sphere_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<sphere_t, sphere_t> >();
      sphere_capsule_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<sphere_t, capsule_t> >();
      sphere_cylinder_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<sphere_t, cylinder_t> >();
      capsule_capsule_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<capsule_t, capsule_t> >();
      capsule_cylinder_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<capsule_t, cylinder_t> >();
      cylinder_cylinder_broadphase_ = allocator_->registerSystem<oy::BroadphaseSystem<cylinder_t, cylinder_t> >();
 
/// sat
      cube_cube_sat_ = allocator_->registerSystem<oy::SatSystem<cube_t, cube_t> >();
      cube_sphere_sat_ = allocator_->registerSystem<oy::SatSystem<cube_t, sphere_t> >();
      cube_capsule_sat_ = allocator_->registerSystem<oy::SatSystem<cube_t, capsule_t> >();
      sphere_sphere_sat_ = allocator_->registerSystem<oy::SatSystem<sphere_t, sphere_t> >();
      sphere_capsule_sat_ = allocator_->registerSystem<oy::SatSystem<sphere_t, capsule_t> >();
      sphere_cylinder_sat_ = allocator_->registerSystem<oy::SatSystem<sphere_t, cylinder_t> >();
      capsule_capsule_sat_ = allocator_->registerSystem<oy::SatSystem<capsule_t, capsule_t> >();

/// gjk
      cube_cylinder_gjk_ = allocator_->registerSystem<cubeCylinderGjk_t>();
      capsule_cylinder_gjk_ = allocator_->registerSystem<capsuleCylinderGjk_t>();
      cylinder_cylinder_gjk_ = allocator_->registerSystem<cylinderCylinderGjk_t>();

/// epa
      cube_cylinder_epa_ = allocator_->registerSystem<cubeCylinderEpa_t>();
      capsule_cylinder_epa_ = allocator_->registerSystem<capsuleCylinderEpa_t>();
      cylinder_cylinder_epa_ = allocator_->registerSystem<cylinderCylinderEpa_t>();

/// contact manifold

      // This marks all of the contact manifolds as inactive, otherwise every
      // contact manifold system marks *every* contact manifold as inactive, so
      // only contact manifolds from cylinder-cylinder collisions survive.
      manifold_jankifier_ = allocator_->registerSystem<ManifoldComponentJankifier>();

      cube_cube_manifold_ = allocator_->registerSystem<cubeCubeManifold_t>();
      cube_sphere_manifold_ = allocator_->registerSystem<cubeSphereManifold_t>();
      cube_capsule_manifold_ = allocator_->registerSystem<cubeCapsuleManifold_t>();
      cube_cylinder_manifold_ = allocator_->registerSystem<cubeCylinderManifold_t>();
      sphere_sphere_manifold_ = allocator_->registerSystem<sphereSphereManifold_t>();
      sphere_capsule_manifold_ = allocator_->registerSystem<sphereCapsuleManifold_t>();
      sphere_cylinder_manifold_ = allocator_->registerSystem<sphereCylinderManifold_t>();
      capsule_capsule_manifold_ = allocator_->registerSystem<capsuleCapsuleManifold_t>();
      capsule_cylinder_manifold_ = allocator_->registerSystem<capsuleCylinderManifold_t>();
      cylinder_cylinder_manifold_ = allocator_->registerSystem<cylinderCylinderManifold_t>();

      constantForceSystem_ = allocator_->registerSystem<oy::ConstantForceCalculator>();
      damperSystem_ = allocator_->registerSystem<oy::VelocityDamperForceCalculator>();
      dragForceSystem_ = allocator_->registerSystem<oy::DragForceCalculator>();
      springSystem_ = allocator_->registerSystem<oy::SpringForceCalculator>();

      dragTorqueSystem_ = allocator_->registerSystem<oy::DragTorqueCalculator>();

      collision_calculator_ = allocator_->registerSystem<oy::InequalityConstraintCalculatorSystem<oy::types::constraintCollision_t> >();
      friction_calculator_ = allocator_->registerSystem<oy::InequalityConstraintCalculatorSystem<oy::types::constraintFriction_t> >();
      torsional_friction_calculator_ = allocator_->registerSystem<oy::InequalityConstraintCalculatorSystem<oy::types::constraintTorsionalFriction_t> >();

      gear_calculator_ = allocator_->registerSystem<oy::EqualityConstraintCalculatorSystem<oy::types::constraintGear_t> >();
      revolute_motor_calculator_ = allocator_->registerSystem<oy::EqualityConstraintCalculatorSystem<oy::types::constraintRevoluteMotor_t> >();
      rotation_1d_calculator_ = allocator_->registerSystem<oy::EqualityConstraintCalculatorSystem<oy::types::constraintRotation1d_t> >();
      translation_1d_calculator_ = allocator_->registerSystem<oy::EqualityConstraintCalculatorSystem<oy::types::constraintTranslation1d_t> >();

      collisionSystem_ = allocator_->registerSystem<oy::CollisionSystem>();
      frictionSystem_ = allocator_->registerSystem<oy::FrictionSystem>();
      torsionalFrictionSystem_ = allocator_->registerSystem<oy::TorsionalFrictionSystem>();
      constrained_body_system_ = allocator_->registerSystem<oy::ConstrainedRigidBodySystem>();
      constraintSolver_ = allocator_->registerSystem<oy::ConstraintSolver>();
   }

   void Handle::registerQueriesAndComponents(void)
   {
      allocator_->registerComponent<oy::types::compositeConstraintJoint_t>();

      allocator_->registerComponent<oy::types::constraintBalljoint_t>();

      allocator_->registerComponent<oy::types::constraintRevoluteJoint_t>();

      rigid_body_query_ = allocator_->addArchetypeQuery<
         oy::types::rigidBody_t,
         oy::types::isometricCollider_t,
         geometry::types::aabb_t
      >();

      dynamic_body_query_ = allocator_->addArchetypeQuery<
         oy::types::rigidBody_t,
         oy::types::isometricCollider_t,
         geometry::types::aabb_t,
         oy::types::DynamicBody
      >();

      stationary_body_query_ = allocator_->addArchetypeQuery<
         oy::types::rigidBody_t,
         oy::types::isometricCollider_t,
         geometry::types::aabb_t,
         oy::types::StationaryBody
      >();

      balljoint_constraint_query_ = \
         allocator_->addArchetypeQuery<oy::types::constraintBalljoint_t>();

      gear_constraint_query_ = \
         allocator_->addArchetypeQuery<oy::types::constraintGear_t>();

      revolute_joint_constraint_query_ = \
         allocator_->addArchetypeQuery<oy::types::constraintRevoluteJoint_t>();

      revolute_motor_constraint_query_ = \
         allocator_->addArchetypeQuery<oy::types::constraintRevoluteMotor_t>();

      rotation_1d_constraint_query_ = \
         allocator_->addArchetypeQuery<oy::types::constraintRotation1d_t>();

      translation_1d_constraint_query_ = \
         allocator_->addArchetypeQuery<oy::types::constraintTranslation1d_t>();

      constant_force_query_ = \
         allocator_->addArchetypeQuery<oy::types::forceConstant_t>();

      drag_force_query_ = \
         allocator_->addArchetypeQuery<oy::types::forceDrag_t>();

      spring_force_query_ = \
         allocator_->addArchetypeQuery<oy::types::forceSpring_t>();

      velocity_damper_force_query_ = \
         allocator_->addArchetypeQuery<oy::types::forceVelocityDamper_t>();

      drag_torque_query_ = \
         allocator_->addArchetypeQuery<oy::types::torqueDrag_t>();

      edge_query_ = allocator_->addArchetypeQuery<trecs::edge_t>();
   }

   void Handle::setupScenario(const oy::types::scenario_t & scenario)
   {
      scenario_to_sim_id_.clear();
      // Make an entry for the null body entity because some constraints use
      // this in lieu of an actual body ID.
      scenario_to_sim_id_[oy::types::null_body_id] = oy::types::null_body_entity;

      // Add bodies to the scenario using the interface in this handle.
      for (
         auto body_it = scenario.bodies.begin();
         body_it != scenario.bodies.end();
         ++body_it
      )
      {
         int body_id = body_it->first;

         const trecs::uid_t temp_body_id = addBody(
            body_it->second,
            scenario.isometric_colliders.at(body_id),
            scenario.shapes.at(body_id),
            scenario.body_types.at(body_id)
         );

         if (scenario.body_types.at(body_it->first) == oy::types::enumRigidBody_t::STATIONARY)
         {
            // Zero-out any time derivative information from stationary bodies.
            oy::types::rigidBody_t & bodyRef = getBody(temp_body_id);
            bodyRef.angVel.Initialize(0.f, 0.f, 0.f);
            bodyRef.linVel.Initialize(0.f, 0.f, 0.f);
         }

         scenario_to_sim_id_[body_id] = temp_body_id;
         std::cout << "scenario id: " << body_id << " body id: " << temp_body_id << "\n";
      }

      // Map scenario balljoint body IDs back to internal body IDs
      // Add the balljoint constraints
      for (
         auto bj_it = scenario.balljoints.begin();
         bj_it != scenario.balljoints.end();
         ++bj_it
      )
      {
         if (
            (scenario_to_sim_id_.find(bj_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(bj_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (bj_it->first.parentId != oy::types::null_body_entity) ||
               (bj_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::constraintBalljoint_t temp_ball_joint = bj_it->second;

            addBalljointConstraint(
               scenario_to_sim_id_[bj_it->first.parentId],
               scenario_to_sim_id_[bj_it->first.childId],
               temp_ball_joint
            );
         }
      }

      // Map scenario gear body IDs back to internal body IDs
      // Add the gear constraints
      for (
         auto ge_it = scenario.gears.begin();
         ge_it != scenario.gears.end();
         ++ge_it
      )
      {
         if (
            (scenario_to_sim_id_.find(ge_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(ge_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (ge_it->first.parentId != oy::types::null_body_entity) &&
               (ge_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::constraintGear_t temp_gear = ge_it->second;

            addGearConstraint(
               scenario_to_sim_id_[ge_it->first.parentId],
               scenario_to_sim_id_[ge_it->first.childId],
               temp_gear
            );
         }
      }

      // Map scenario revolute joint body IDs back to internal body IDs
      // Add the revolute joint constraints
      for (
         auto rj_it = scenario.revolute_joints.begin();
         rj_it != scenario.revolute_joints.end();
         ++rj_it
      )
      {
         if (
            (scenario_to_sim_id_.find(rj_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(rj_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (rj_it->first.parentId != oy::types::null_body_entity) ||
               (rj_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::constraintRevoluteJoint_t temp_revolute_joint = rj_it->second;

            addRevoluteJointConstraint(
               scenario_to_sim_id_[rj_it->first.parentId],
               scenario_to_sim_id_[rj_it->first.childId],
               temp_revolute_joint
            );
         }
      }

      // Map scenario revolute motor body IDs back to internal body IDs
      // Add the revolute motor constraints
      for (
         auto rm_it = scenario.revolute_motors.begin();
         rm_it != scenario.revolute_motors.end();
         ++rm_it
      )
      {
         if (
            (scenario_to_sim_id_.find(rm_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(rm_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (rm_it->first.parentId != oy::types::null_body_entity) ||
               (rm_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::constraintRevoluteMotor_t temp_revolute_motor = rm_it->second;

            addRevoluteMotorConstraint(
               scenario_to_sim_id_[rm_it->first.parentId],
               scenario_to_sim_id_[rm_it->first.childId],
               temp_revolute_motor
            );
         }
      }

      for (
         auto ro_it = scenario.rotation_1d.begin();
         ro_it != scenario.rotation_1d.end();
         ++ro_it
      )
      {
         if (
            (scenario_to_sim_id_.find(ro_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(ro_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (ro_it->first.parentId != oy::types::null_body_entity) ||
               (ro_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::constraintRotation1d_t temp_rotation_1d = ro_it->second;

            addRotation1dConstraint(
               scenario_to_sim_id_[ro_it->first.parentId],
               scenario_to_sim_id_[ro_it->first.childId],
               temp_rotation_1d
            );
         }
      }

      for (
         auto tr_it = scenario.translation_1d.begin();
         tr_it != scenario.translation_1d.end();
         ++tr_it
      )
      {
         if (
            (scenario_to_sim_id_.find(tr_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(tr_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (tr_it->first.parentId != oy::types::null_body_entity) ||
               (tr_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::constraintTranslation1d_t temp_translation_1d = tr_it->second;

            addTranslation1dConstraint(
               scenario_to_sim_id_[tr_it->first.parentId],
               scenario_to_sim_id_[tr_it->first.childId],
               temp_translation_1d
            );
         }
      }

      for (
         auto cf_it = scenario.constant_forces.begin();
         cf_it != scenario.constant_forces.end();
         ++cf_it
      )
      {
         if (
            (scenario_to_sim_id_.find(cf_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(cf_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (cf_it->first.parentId != oy::types::null_body_entity) ||
               (cf_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::forceConstant_t temp_constant_force = cf_it->second;

            addConstantForce(
               scenario_to_sim_id_[cf_it->first.childId],
               temp_constant_force
            );
         }
      }

      for (
         auto df_it = scenario.drag_forces.begin();
         df_it != scenario.drag_forces.end();
         ++df_it
      )
      {
         if (
            (scenario_to_sim_id_.find(df_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(df_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (df_it->first.parentId != oy::types::null_body_entity) ||
               (df_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::forceDrag_t temp_drag_force = df_it->second;

            addDragForce(
               scenario_to_sim_id_[df_it->first.childId],
               temp_drag_force
            );
         }
      }

      for (
         auto sp_it = scenario.springs.begin();
         sp_it != scenario.springs.end();
         ++sp_it
      )
      {
         if (
            (scenario_to_sim_id_.find(sp_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(sp_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (sp_it->first.parentId != oy::types::null_body_entity) ||
               (sp_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::forceSpring_t temp_spring = sp_it->second;

            addSpringForce(
               scenario_to_sim_id_[sp_it->first.parentId],
               scenario_to_sim_id_[sp_it->first.childId],
               temp_spring
            );
         }
      }

      for (
         auto da_it = scenario.dampers.begin();
         da_it != scenario.dampers.end();
         ++da_it
      )
      {
         if (
            (scenario_to_sim_id_.find(da_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(da_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (da_it->first.parentId != oy::types::null_body_entity) ||
               (da_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::forceVelocityDamper_t temp_damper = da_it->second;

            addVelocityDamperForce(
               scenario_to_sim_id_[da_it->first.parentId],
               scenario_to_sim_id_[da_it->first.childId],
               temp_damper
            );
         }
      }

      for (
         auto dt_it = scenario.drag_torques.begin();
         dt_it != scenario.drag_torques.end();
         ++dt_it
      )
      {
         if (
            (scenario_to_sim_id_.find(dt_it->first.parentId) != scenario_to_sim_id_.end()) &&
            (scenario_to_sim_id_.find(dt_it->first.childId) != scenario_to_sim_id_.end()) &&
            (
               (dt_it->first.parentId != oy::types::null_body_entity) ||
               (dt_it->first.childId != oy::types::null_body_entity)
            )
         )
         {
            const oy::types::torqueDrag_t temp_drag_torque = dt_it->second;

            addDragTorque(
               scenario_to_sim_id_[dt_it->first.childId],
               temp_drag_torque
            );
         }
      }

      logger::types::loggerConfig_t logger_config = scenario.logger;
      // No good way to do this.
      // logger_config.frameCounter = &oy::getFrameCount;
      // logger_config.frameCounter = &(this->allocator_->getFrameCounter);

      logger::setLoggerConfig(logger_config);
   }

   void Handle::step(void)
   {
      cube_aabb_system_->update(*allocator_);
      sphere_aabb_system_->update(*allocator_);
      capsule_aabb_system_->update(*allocator_);
      cylinder_aabb_system_->update(*allocator_);

      cube_cube_broadphase_->update(*allocator_);
      cube_sphere_broadphase_->update(*allocator_);
      cube_capsule_broadphase_->update(*allocator_);
      cube_cylinder_broadphase_->update(*allocator_);
      sphere_sphere_broadphase_->update(*allocator_);
      sphere_capsule_broadphase_->update(*allocator_);
      sphere_cylinder_broadphase_->update(*allocator_);
      capsule_capsule_broadphase_->update(*allocator_);
      capsule_cylinder_broadphase_->update(*allocator_);
      cylinder_cylinder_broadphase_->update(*allocator_);

      cube_cylinder_gjk_->update(*allocator_);
      capsule_cylinder_gjk_->update(*allocator_);
      cylinder_cylinder_gjk_->update(*allocator_);

      cube_cube_sat_->update(*allocator_);
      cube_sphere_sat_->update(*allocator_);
      cube_capsule_sat_->update(*allocator_);
      sphere_sphere_sat_->update(*allocator_);
      sphere_capsule_sat_->update(*allocator_);
      sphere_cylinder_sat_->update(*allocator_);
      capsule_capsule_sat_->update(*allocator_);

      cube_cylinder_epa_->update(*allocator_);
      capsule_cylinder_epa_->update(*allocator_);
      cylinder_cylinder_epa_->update(*allocator_);

      // Clears the vector of manifolds.
      manifold_jankifier_->update(*allocator_);

      cube_cube_manifold_->update(*allocator_);
      cube_sphere_manifold_->update(*allocator_);
      cube_capsule_manifold_->update(*allocator_);
      cube_cylinder_manifold_->update(*allocator_);
      sphere_sphere_manifold_->update(*allocator_);
      sphere_capsule_manifold_->update(*allocator_);
      sphere_cylinder_manifold_->update(*allocator_);
      capsule_capsule_manifold_->update(*allocator_);
      capsule_cylinder_manifold_->update(*allocator_);
      cylinder_cylinder_manifold_->update(*allocator_);

      springSystem_->update(*allocator_);
      damperSystem_->update(*allocator_);
      dragForceSystem_->update(*allocator_);
      constantForceSystem_->update(*allocator_);

      dragTorqueSystem_->update(*allocator_);

      constrained_body_system_->update(*allocator_);

      collisionSystem_->update(*allocator_);
      frictionSystem_->update(*allocator_);
      torsionalFrictionSystem_->update(*allocator_);

      collision_calculator_->update(*allocator_);
      friction_calculator_->update(*allocator_);
      torsional_friction_calculator_->update(*allocator_);

      gear_calculator_->update(*allocator_);
      revolute_motor_calculator_->update(*allocator_);
      rotation_1d_calculator_->update(*allocator_);
      translation_1d_calculator_->update(*allocator_);

      constraintSolver_->update(*allocator_);

      rk4System_->update(*allocator_);

      ++frame_counter_;
   }

   void Handle::applyForce(
      const trecs::uid_t body_id,
      const Vector3 force,
      const oy::types::enumFrame_t frame
   )
   {
      auto body = allocator_->getComponent<oy::types::rigidBody_t>(body_id);
      auto forque = allocator_->getComponent<oy::types::generalizedForce_t>(body_id);

      if (body != nullptr && forque != nullptr)
      {
         oy::rb::applyForce(
            *body, force, frame, *forque
         );
      }
   }

   trecs::uid_t Handle::addBody(
      const oy::types::rigidBody_t & rigid_body,
      const oy::types::isometricCollider_t & collider,
      const geometry::types::shape_t & shape,
      const oy::types::enumRigidBody_t body_type
   )
   {
      trecs::uid_t body_uid = allocator_->addEntity();
      if (body_uid == oy::types::null_body_entity)
      {
         return body_uid;
      }

      allocator_->addComponent(body_uid, rigid_body);
      allocator_->addComponent(body_uid, collider);
      allocator_->addComponent(body_uid, shape);
      allocator_->addComponent(body_uid, oy::types::generalizedForce_t{});
      allocator_->addComponent(body_uid, oy::types::rk4Midpoint_t{});

      switch(body_type)
      {
         case oy::types::enumRigidBody_t::DYNAMIC:
         {
            allocator_->addComponent(body_uid, oy::types::DynamicBody{});
            allocator_->addComponent(body_uid, oy::types::rk4Increment_t<0>{});
            allocator_->addComponent(body_uid, oy::types::rk4Increment_t<1>{});
            allocator_->addComponent(body_uid, oy::types::rk4Increment_t<2>{});
            allocator_->addComponent(body_uid, oy::types::rk4Increment_t<3>{});
            break;
         }
         case oy::types::enumRigidBody_t::STATIONARY:
         {

            allocator_->addComponent(body_uid, oy::types::StationaryBody{});
            break;
         }
      }

      switch(shape.shapeType)
      {
         case geometry::types::enumShape_t::CUBE:
         {
            if (!allocator_->addComponent(body_uid, shape.cube))
            {
               std::cout << "Couldn't add cube to body entity " << body_uid << "\n";
            }
            break;
         }
         case geometry::types::enumShape_t::SPHERE:
         {
            if (!allocator_->addComponent(body_uid, shape.sphere))
            {
               std::cout << "Couldn't add sphere to body entity " << body_uid << "\n";
            }
            break;
         }
         case geometry::types::enumShape_t::CAPSULE:
         {
            if (!allocator_->addComponent(body_uid, shape.capsule))
            {
               std::cout << "Couldn't add capsule to body entity " << body_uid << "\n";
            }
            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            if (!allocator_->addComponent(body_uid, shape.cylinder))
            {
               std::cout << "Couldn't add cylinder to body entity " << body_uid << "\n";
            }
            break;
         }
         case geometry::types::enumShape_t::NONE:
         {
            allocator_->addComponent(
               body_uid, geometry::types::shapeSphere_t{1.f}
            );
            break;
         }
      }

      // Just has to exist, doesn't have to be correct. Broadphase will change
      // it anyway.
      geometry::types::aabb_t temp_aabb;
      allocator_->addComponent(body_uid, temp_aabb);

      return body_uid;
   }

   oy::types::rigidBody_t & Handle::getBody(trecs::uid_t entity)
   {
      oy::types::rigidBody_t * body = \
         allocator_->getComponent<oy::types::rigidBody_t>(entity);

      if (body == nullptr)
      {
         std::string error_string("Couldn't find rigid body at entity ");
         error_string += std::to_string(entity);

         throw std::invalid_argument(error_string);
      }

      return *body;
   }

   bool Handle::stationary(const trecs::uid_t entity) const
   {
      return !allocator_->hasComponent<oy::types::DynamicBody>(entity);
   }

   const std::unordered_set<trecs::uid_t> & Handle::getBodyUids(void) const
   {
      return allocator_->getQueryEntities(rigid_body_query_);
   }

   oy::types::isometricCollider_t & Handle::getCollider(trecs::uid_t entity)
   {
      return *(allocator_->getComponent<oy::types::isometricCollider_t>(entity));
   }

   geometry::types::aabb_t & Handle::getAabb(trecs::uid_t entity)
   {
      return *(allocator_->getComponent<geometry::types::aabb_t>(entity));
   }

   geometry::types::shape_t & Handle::getShape(trecs::uid_t entity)
   {
      return *(allocator_->getComponent<geometry::types::shape_t>(entity));
   }

   oy::types::bodyLink_t Handle::getBodyLink(trecs::uid_t entity) const
   {
      trecs::edge_t * temp_link = allocator_->getComponent<trecs::edge_t>(entity);

      if (temp_link == nullptr)
      {
         return {0, 0};
      }

      return {temp_link->nodeIdA, temp_link->nodeIdB};
   }

   trecs::uid_t Handle::addBalljointConstraint(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::constraintBalljoint_t & constraint
   )
   {
      const trecs::uid_t balljoint_entity = addCompositeLinkComponent(
         parent_id, child_id, constraint, "ball joint"
      );

      oy::types::compositeConstraintJoint_t * composite_joint = \
         allocator_->getComponent<oy::types::compositeConstraintJoint_t>(
            balljoint_entity
         );

      composite_joint->numRotationConstraints = 0;
      composite_joint->numTranslationConstraints = 3;

      for (int i = 0; i < 3; ++i)
      {
         oy::types::constraintTranslation1d_t temp_trans_cons;
         temp_trans_cons.parentAxis[i] = 1.f;
         temp_trans_cons.parentLinkPoint = constraint.parentLinkPoint;
         temp_trans_cons.childLinkPoint = constraint.childLinkPoint;

         composite_joint->translationConstraintIds[i] = addTranslation1dConstraint(
            parent_id, child_id, temp_trans_cons
         );
      }

      return balljoint_entity;
   }

   const std::unordered_set<trecs::uid_t> & Handle::getBalljointConstraintUids(void) const
   {
      return allocator_->getQueryEntities(balljoint_constraint_query_);
   }

   const oy::types::constraintBalljoint_t & Handle::getBalljointConstraint(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::constraintBalljoint_t>(
         entity, "ball joint"
      );
   }

   void Handle::setBalljointConstraint(
      const trecs::uid_t entity,
      const oy::types::constraintBalljoint_t & constraint
   )
   {
      if (!allocator_->hasComponent<oy::types::constraintBalljoint_t>(entity))
      {
         return;
      }

      getLinkComponent<oy::types::constraintBalljoint_t>(entity, "ball joint") = constraint;

      oy::types::compositeConstraintJoint_t * composite_joint = \
         allocator_->getComponent<oy::types::compositeConstraintJoint_t>(
            entity
         );

      composite_joint->numRotationConstraints = 0;
      composite_joint->numTranslationConstraints = 3;

      for (int i = 0; i < 3; ++i)
      {
         oy::types::constraintTranslation1d_t * temp_trans_cons = \
            allocator_->getComponent<oy::types::constraintTranslation1d_t>(
               composite_joint->translationConstraintIds[i]
            );
         temp_trans_cons->parentAxis[i] = 1.f;
         temp_trans_cons->parentLinkPoint = constraint.parentLinkPoint;
         temp_trans_cons->childLinkPoint = constraint.childLinkPoint;
      }
   }

   void Handle::removeBalljointConstraint(const trecs::uid_t entity)
   {
      removeCompositeLinkComponent<oy::types::constraintBalljoint_t>(entity);
   }

   trecs::uid_t Handle::addGearConstraint(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::constraintGear_t & constraint
   )
   {
      return addLinkComponent(parent_id, child_id, constraint, "gear");
   }

   const std::unordered_set<trecs::uid_t> & Handle::getGearConstraintUids(void) const
   {
      return allocator_->getQueryEntities(gear_constraint_query_);
   }

   oy::types::constraintGear_t & Handle::getGearConstraint(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::constraintGear_t>(
         entity, "gear"
      );
   }

   const oy::types::constraintGear_t & Handle::getGearConstraint(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::constraintGear_t>(
         entity, "gear"
      );
   }

   trecs::uid_t Handle::addRevoluteJointConstraint(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::constraintRevoluteJoint_t & constraint
   )
   {
      const trecs::uid_t revolute_joint_entity = addCompositeLinkComponent(
         parent_id, child_id, constraint, "revolute joint"
      );

      oy::types::compositeConstraintJoint_t * composite_joint = \
         allocator_->getComponent<oy::types::compositeConstraintJoint_t>(
            revolute_joint_entity
         );

      composite_joint->numRotationConstraints = 2;
      composite_joint->numTranslationConstraints = 3;

      const Vector3 parent_link_point = (
         constraint.parentLinkPoints[0] + constraint.parentLinkPoints[1]
      ) / 2.f;

      const Vector3 child_link_point = (
         constraint.childLinkPoints[0] + constraint.childLinkPoints[1]
      ) / 2.f;

      for (int i = 0; i < 3; ++i)
      {
         oy::types::constraintTranslation1d_t temp_trans_cons;
         temp_trans_cons.parentAxis[i] = 1.f;
         temp_trans_cons.parentLinkPoint = parent_link_point;
         temp_trans_cons.childLinkPoint = child_link_point;

         composite_joint->translationConstraintIds[i] = addTranslation1dConstraint(
            parent_id, child_id, temp_trans_cons
         );
      }

      const Vector3 parent_link_axis_A = (
         constraint.parentLinkPoints[1] - constraint.parentLinkPoints[0]
      ).unitVector();

      const Vector3 child_link_axis_B = (
         constraint.childLinkPoints[1] - constraint.childLinkPoints[0]
      ).unitVector();

      // Generates a rotation matrix from child frame to child axis frame where
      // the child link axis is in the z-hat direction.
      const Matrix33 R_child_to_axis = makeVectorUp(child_link_axis_B);

      for (int i = 0; i < 2; ++i)
      {
         // Pick the arbitrarily generated x-hat and y-hat axis in child link
         // axis frame as the vectors that'll be orthogonal to the parent link
         // axis.
         Vector3 axis_vector_C;
         axis_vector_C[i] = 1.f;

         oy::types::constraintRotation1d_t temp_rot_cons;
         temp_rot_cons.parentAxis = parent_link_axis_A;
         temp_rot_cons.childAxis = R_child_to_axis.transpose() * axis_vector_C;

         composite_joint->rotationConstraintIds[i] = addRotation1dConstraint(
            parent_id, child_id, temp_rot_cons
         );
      }

      return revolute_joint_entity;
   }

   const std::unordered_set<trecs::uid_t> & Handle::getRevoluteJointConstraintUids(void) const
   {
      return allocator_->getQueryEntities(revolute_joint_constraint_query_);
   }

   const oy::types::constraintRevoluteJoint_t & Handle::getRevoluteJointConstraint(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::constraintRevoluteJoint_t>(
         entity, "revolute joint"
      );
   }

   void Handle::setRevoluteJointConstraint(
      const trecs::uid_t entity,
      const oy::types::constraintRevoluteJoint_t & constraint
   )
   {
      if (!allocator_->hasComponent<oy::types::constraintRevoluteJoint_t>(entity))
      {
         return;
      }

      getLinkComponent<oy::types::constraintRevoluteJoint_t>(entity, "revolute joint") = constraint;

      oy::types::compositeConstraintJoint_t * composite_joint = \
         allocator_->getComponent<oy::types::compositeConstraintJoint_t>(
            entity
         );

      composite_joint->numRotationConstraints = 2;
      composite_joint->numTranslationConstraints = 3;

      const Vector3 parent_link_point = (
         constraint.parentLinkPoints[0] + constraint.parentLinkPoints[1]
      ) / 2.f;

      const Vector3 child_link_point = (
         constraint.childLinkPoints[0] + constraint.childLinkPoints[1]
      ) / 2.f;

      for (int i = 0; i < 3; ++i)
      {
         oy::types::constraintTranslation1d_t * temp_trans_cons = \
            allocator_->getComponent<oy::types::constraintTranslation1d_t>(
               composite_joint->translationConstraintIds[i]
            );
         temp_trans_cons->parentAxis[i] = 1.f;
         temp_trans_cons->parentLinkPoint = parent_link_point;
         temp_trans_cons->childLinkPoint = child_link_point;
      }

      const Vector3 parent_link_axis_A = (
         constraint.parentLinkPoints[1] - constraint.parentLinkPoints[0]
      ).unitVector();

      const Vector3 child_link_axis_B = (
         constraint.childLinkPoints[1] - constraint.childLinkPoints[0]
      ).unitVector();

      // Generates a rotation matrix from child frame to child axis frame where
      // the child link axis is in the z-hat direction.
      const Matrix33 R_child_to_axis = makeVectorUp(child_link_axis_B);

      for (int i = 0; i < 2; ++i)
      {
         oy::types::constraintRotation1d_t * temp_rot_cons = \
            allocator_->getComponent<oy::types::constraintRotation1d_t>(
               composite_joint->rotationConstraintIds[i]
            );

         // Pick the arbitrarily generated x-hat and y-hat axis in child link
         // axis frame as the vectors that'll be orthogonal to the parent link
         // axis.
         Vector3 axis_vector_C;
         axis_vector_C[i] = 1.f;

         temp_rot_cons->parentAxis = parent_link_axis_A;
         temp_rot_cons->childAxis = R_child_to_axis.transpose() * axis_vector_C;
      }

   }

   void Handle::removeRevoluteJointConstraint(const trecs::uid_t entity)
   {
      removeCompositeLinkComponent<oy::types::constraintRevoluteJoint_t>(entity);
   }

   trecs::uid_t Handle::addRevoluteMotorConstraint(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::constraintRevoluteMotor_t & constraint
   )
   {
      return addLinkComponent(parent_id, child_id, constraint, "revolute motor");
   }

   const std::unordered_set<trecs::uid_t> & Handle::getRevoluteMotorConstraintUids(void) const
   {
      return allocator_->getQueryEntities(revolute_motor_constraint_query_);
   }

   oy::types::constraintRevoluteMotor_t & Handle::getRevoluteMotorConstraint(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::constraintRevoluteMotor_t>(
         entity, "revolute motor"
      );
   }

   const oy::types::constraintRevoluteMotor_t & Handle::getRevoluteMotorConstraint(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::constraintRevoluteMotor_t>(
         entity, "revolute motor"
      );
   }

   // Adds a translation 1D constraint (equality) between two bodies.
   trecs::uid_t Handle::addRotation1dConstraint(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::constraintRotation1d_t & constraint
   )
   {
      return addLinkComponent(parent_id, child_id, constraint, "1d rotation");
   }

   const std::unordered_set<trecs::uid_t> & Handle::getRotation1dConstraintUids(void) const
   {
      return allocator_->getQueryEntities(rotation_1d_constraint_query_);
   }

   oy::types::constraintRotation1d_t & Handle::getRotation1dConstraint(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::constraintRotation1d_t>(
         entity, "1d rotation"
      );
   }

   const oy::types::constraintRotation1d_t & Handle::getRotation1dConstraint(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::constraintRotation1d_t>(
         entity, "1d rotation"
      );
   }

   // Adds a translation 1D constraint (equality) between two bodies.
   trecs::uid_t Handle::addTranslation1dConstraint(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::constraintTranslation1d_t & constraint
   )
   {
      return addLinkComponent(parent_id, child_id, constraint, "1d translation");
   }

   const std::unordered_set<trecs::uid_t> & Handle::getTranslation1dConstraintUids(void) const
   {
      return allocator_->getQueryEntities(translation_1d_constraint_query_);
   }

   oy::types::constraintTranslation1d_t & Handle::getTranslation1dConstraint(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::constraintTranslation1d_t>(
         entity, "1d translation"
      );
   }

   const oy::types::constraintTranslation1d_t & Handle::getTranslation1dConstraint(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::constraintTranslation1d_t>(
         entity, "1d translation"
      );
   }

   void Handle::removeEntity(const trecs::uid_t entity)
   {
      allocator_->removeEntity(entity);
   }

   trecs::uid_t Handle::addConstantForce(
      const trecs::uid_t child_id,
      const oy::types::forceConstant_t & force
   )
   {
      return addLinkComponent<oy::types::forceConstant_t>(
         oy::types::null_body_entity, child_id, force, "constant force"
      );
   }

   const std::unordered_set<trecs::uid_t> & Handle::getConstantForceUids(void) const
   {
      return allocator_->getQueryEntities(constant_force_query_);
   }

   oy::types::forceConstant_t & Handle::getConstantForce(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::forceConstant_t>(
         entity, "constant force"
      );
   }

   const oy::types::forceConstant_t & Handle::getConstantForce(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::forceConstant_t>(
         entity, "constant force"
      );
   }

   trecs::uid_t Handle::addDragForce(
      const trecs::uid_t child_id,
      const oy::types::forceDrag_t & force
   )
   {
      return addLinkComponent<oy::types::forceDrag_t>(
         oy::types::null_body_entity, child_id, force, "drag force"
      );
   }

   const std::unordered_set<trecs::uid_t> & Handle::getDragForceUids(void) const
   {
      return allocator_->getQueryEntities(drag_force_query_);
   }

   oy::types::forceDrag_t & Handle::getDragForce(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::forceDrag_t>(
         entity, "drag force"
      );
   }

   const oy::types::forceDrag_t & Handle::getDragForce(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::forceDrag_t>(
         entity, "drag force"
      );
   }

   trecs::uid_t Handle::addSpringForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceSpring_t & force
   )
   {
      return addLinkComponent<oy::types::forceSpring_t>(
         parent_id, child_id, force, "spring"
      );
   }

   oy::types::forceSpring_t & Handle::getSpringForce(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::forceSpring_t>(
         entity, "spring"
      );
   }

   const oy::types::forceSpring_t & Handle::getSpringForce(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::forceSpring_t>(
         entity, "spring"
      );
   }

   trecs::uid_t Handle::addVelocityDamperForce(
      const trecs::uid_t parent_id,
      const trecs::uid_t child_id,
      const oy::types::forceVelocityDamper_t & force
   )
   {
      return addLinkComponent<oy::types::forceVelocityDamper_t>(
         parent_id, child_id, force, "velocity damper"
      );
   }

   oy::types::forceVelocityDamper_t & Handle::getVelocityDamperForce(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::forceVelocityDamper_t>(
         entity, "velocity damper"
      );
   }

   const oy::types::forceVelocityDamper_t & Handle::getVelocityDamperForce(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::forceVelocityDamper_t>(
         entity, "velocity damper"
      );
   }

   trecs::uid_t Handle::addDragTorque(
      const trecs::uid_t child_id,
      const oy::types::torqueDrag_t & force
   )
   {
      return addLinkComponent<oy::types::torqueDrag_t>(
         oy::types::null_body_entity, child_id, force, "drag torque"
      );
   }

   const std::unordered_set<trecs::uid_t> & Handle::getDragTorqueUids(void) const
   {
      return allocator_->getQueryEntities(drag_torque_query_);
   }

   oy::types::torqueDrag_t & Handle::getDragTorque(
      const trecs::uid_t entity
   )
   {
      return getLinkComponent<oy::types::torqueDrag_t>(
         entity, "drag torque"
      );
   }

   const oy::types::torqueDrag_t & Handle::getDragTorque(
      const trecs::uid_t entity
   ) const
   {
      return getLinkComponent<oy::types::torqueDrag_t>(
         entity, "drag torque"
      );
   }

   oy::types::raycastResult_t Handle::raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_unit,
      float max_distance,
      oy::types::enumRaycastBodyFilter_t filter_type
   ) const
   {
      float closest_distance = __FLT_MAX__;
      trecs::uid_t closest_uid = oy::types::null_body_entity;

      oy::types::raycastResult_t result;
      result.hit = false;
      result.bodyId = oy::types::null_body_entity;

      std::unordered_set<trecs::uid_t> entities;

      switch(filter_type)
      {
         case oy::types::enumRaycastBodyFilter_t::ALL:
         {
            entities = allocator_->getQueryEntities(rigid_body_query_);
            break;
         }
         case oy::types::enumRaycastBodyFilter_t::DYNAMIC:
         {
            entities = allocator_->getQueryEntities(dynamic_body_query_);
            break;
         }
         case oy::types::enumRaycastBodyFilter_t::STATIONARY:
         {
            entities = allocator_->getQueryEntities(stationary_body_query_);
            break;
         }
      }

      for (const auto entity : entities)
      {
         const oy::types::rigidBody_t * temp_body = \
            allocator_->getComponent<oy::types::rigidBody_t>(entity);

         if (temp_body == nullptr)
         {
            continue;
         }

         const geometry::types::aabb_t * aabb = \
            allocator_->getComponent<geometry::types::aabb_t>(entity);

         if (aabb == nullptr)
         {
            continue;
         }

         Vector3 intersection_point;
         bool aabb_hit = geometry::mesh::rayIntersect(
            *aabb, ray_start, ray_unit, intersection_point
         );

         if (!aabb_hit)
         {
            continue;
         }

         const geometry::types::shape_t * shape = \
            allocator_->getComponent<geometry::types::shape_t>(entity);

         geometry::types::isometricTransform_t trans_Co_to_W = getColliderTransform(entity);

         const Vector3 ray_end = ray_start + ray_unit;

         geometry::types::raycastResult_t raycast_result = \
            geometry::raycast(ray_start, ray_end, trans_Co_to_W, *shape);

         float hit_distance = (ray_start - raycast_result.hits[0]).magnitude();
         if (raycast_result.hit && hit_distance < max_distance)
         {
            if (hit_distance < closest_distance)
            {
               closest_distance = hit_distance;
               closest_uid = static_cast<int>(entity);

               result.bodyId = closest_uid;
               for (unsigned int i = 0; i < raycast_result.numHits; ++i)
               {
                  result.hits[i] = raycast_result.hits[i];
               }
               result.hit = true;
               result.numHits = raycast_result.numHits;
            }
         }
      }

      return result;
   }

   geometry::types::isometricTransform_t Handle::getColliderTransform(
      trecs::uid_t body_uid
   ) const
   {
      oy::types::rigidBody_t * body = \
         allocator_->getComponent<oy::types::rigidBody_t>(body_uid);

      geometry::types::isometricTransform_t trans_Co_to_W = {
         identityMatrix(), {0.f, 0.f, 0.f}
      };

      if (body == nullptr)
      {
         std::cout << "Couldn't find body or collider at id: " << body_uid << "\n";
         return trans_Co_to_W;
      }

      trans_Co_to_W.rotate = body->ql2b.rotationMatrix().transpose();
      trans_Co_to_W.translate = body->linPos;

      return trans_Co_to_W;
   }

   geometry::types::transform_t Handle::getBodyTransform(trecs::uid_t body_uid) const
   {
      oy::types::rigidBody_t * body = \
         allocator_->getComponent<oy::types::rigidBody_t>(body_uid);

      geometry::types::transform_t trans_Bo_to_W = {
         identityMatrix(), identityMatrix(), {0.f, 0.f, 0.f}
      };

      if (body == nullptr)
      {
         std::cout << "Couldn't find body at id: " << body_uid << "\n";
         return trans_Bo_to_W;
      }

      trans_Bo_to_W = {
         identityMatrix(),
         body->ql2b.rotationMatrix().transpose(),
         body->linPos
      };

      return trans_Bo_to_W;
   }

   unsigned int Handle::getFrameCount(void)
   {
      return frame_counter_;
   }

   float Handle::dt(void) const
   {
      return oy::integrator::dt_;
   }

   const std::map<int, trecs::uid_t> & Handle::getIdMapping(void)
   {
      return scenario_to_sim_id_;
   }

}
