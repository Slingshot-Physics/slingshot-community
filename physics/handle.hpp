#ifndef SLINGSHOT_HANDLE_HEADER
#define SLINGSHOT_HANDLE_HEADER

#include "allocator.hpp"
#include "data_model.h"
#include "slingshot_types.hpp"
#include "geometry_types.hpp"
#include "mesh.hpp"
#include "rigidbody.hpp"

// Systems
#include "aabb_calculator.hpp"
#include "broadphase_system.hpp"
#include "collision_geometry_system.hpp"
#include "collision_manifold_system.hpp"
#include "collision_system.hpp"
#include "constant_force_calculator.hpp"
#include "constrained_rigidbody_system.hpp"
#include "constraint_solver.hpp"
#include "drag_force_calculator.hpp"
#include "drag_torque_calculator.hpp"
#include "equality_constraint_calculator_system.hpp"
#include "friction_system.hpp"
#include "generalized_force_reset_system.hpp"
#include "inequality_constraint_calculator_system.hpp"
#include "manifold_component_jankifier.hpp"
#include "narrowphase_system.hpp"
#include "rk4_finalizer_system.hpp"
#include "rk4_increment_calculator_system.hpp"
#include "rk4_integrator_system.hpp"
#include "rk4_midpoint_calculator_system.hpp"
#include "sat_system.hpp"
#include "spring_force_calculator.hpp"
#include "torsional_friction_system.hpp"
#include "velocity_damper_force_calculator.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace oy
{

   class Handle
   {
      public:
         Handle(void);

         Handle(const data_scenario_t & scenario);

         ~Handle(void);

         // Configures the underlying simulation with a given scenario description.
         // Generates a mapping between the scenario IDs and the IDs actually used
         // in the simulation as an output arg.
         void setupScenario(const oy::types::scenario_t & scenario);

         // Move the simulation forward by one timestep.
         void step(void);

         // Set the timestep.
         void setTimestep(float dt);

         void applyForce(
            const trecs::uid_t body_id,
            const Vector3 force,
            const oy::types::enumFrame_t frame
         );

         // Add a body and a mesh for rendering and collision detection. Body
         // UID doesn't need to be specified in the argument types, one will be
         // provided automatically.
         trecs::uid_t addBody(
            const oy::types::rigidBody_t & rigid_body,
            const oy::types::isometricCollider_t & collider,
            const geometry::types::shape_t & shape,
            const oy::types::enumRigidBody_t body_type
         );

         // Retrieve the entire list of active body IDs.
         const std::unordered_set<trecs::uid_t> & getBodyUids(void) const;

         oy::types::rigidBody_t & getBody(const trecs::uid_t entity);

         // Returns false if the entity has a 'DynamicBody' component, returns
         // true otherwise.
         bool stationary(const trecs::uid_t entity) const;

         oy::types::isometricCollider_t & getCollider(trecs::uid_t entity);

         geometry::types::aabb_t & getAabb(trecs::uid_t entity);

         geometry::types::shape_t & getShape(trecs::uid_t entity);

         oy::types::bodyLink_t getBodyLink(const trecs::uid_t entity) const;

         // Adds a balljoint constraint (equality) between one or two bodies.
         trecs::uid_t addBalljointConstraint(
            const trecs::uid_t parent_id,
            const trecs::uid_t child_id,
            const oy::types::constraintBalljoint_t & constraint
         );

         const std::unordered_set<trecs::uid_t> & getBalljointConstraintUids(void) const;

         const oy::types::constraintBalljoint_t & getBalljointConstraint(
            const trecs::uid_t entity
         ) const;

         void setBalljointConstraint(
            const trecs::uid_t entity,
            const oy::types::constraintBalljoint_t & constraint
         );

         void removeBalljointConstraint(const trecs::uid_t entity);

         // Adds a gear constraint (equality) between two bodies.
         trecs::uid_t addGearConstraint(
            const trecs::uid_t parent_id,
            const trecs::uid_t child_id,
            const oy::types::constraintGear_t & constraint
         );

         const std::unordered_set<trecs::uid_t> & getGearConstraintUids(void) const;

         oy::types::constraintGear_t & getGearConstraint(
            const trecs::uid_t entity
         );

         const oy::types::constraintGear_t & getGearConstraint(
            const trecs::uid_t entity
         ) const;

         // Adds a revolute joint constraint (equality) between one or two bodies.
         trecs::uid_t addRevoluteJointConstraint(
            const trecs::uid_t parent_id,
            const trecs::uid_t child_id,
            const oy::types::constraintRevoluteJoint_t & constraint
         );

         const std::unordered_set<trecs::uid_t> & getRevoluteJointConstraintUids(void) const;

         const oy::types::constraintRevoluteJoint_t & getRevoluteJointConstraint(
            const trecs::uid_t entity
         ) const;

         void setRevoluteJointConstraint(
            const trecs::uid_t entity,
            const oy::types::constraintRevoluteJoint_t & constraint
         );

         void removeRevoluteJointConstraint(const trecs::uid_t entity);

         // Adds a revolute motor constraint (inequality) between one or two bodies.
         trecs::uid_t addRevoluteMotorConstraint(
            const trecs::uid_t parent_id,
            const trecs::uid_t child_id,
            const oy::types::constraintRevoluteMotor_t & constraint
         );

         const std::unordered_set<trecs::uid_t> & getRevoluteMotorConstraintUids(void) const;

         oy::types::constraintRevoluteMotor_t & getRevoluteMotorConstraint(
            const trecs::uid_t entity
         );

         const oy::types::constraintRevoluteMotor_t & getRevoluteMotorConstraint(
            const trecs::uid_t entity
         ) const;

         // Adds a translation 1D constraint (equality) between two bodies.
         trecs::uid_t addRotation1dConstraint(
            const trecs::uid_t parent_id,
            const trecs::uid_t child_id,
            const oy::types::constraintRotation1d_t & constraint
         );

         const std::unordered_set<trecs::uid_t> & getRotation1dConstraintUids(void) const;

         oy::types::constraintRotation1d_t & getRotation1dConstraint(
            const trecs::uid_t entity
         );

         const oy::types::constraintRotation1d_t & getRotation1dConstraint(
            const trecs::uid_t entity
         ) const;

         // Adds a translation 1D constraint (equality) between two bodies.
         trecs::uid_t addTranslation1dConstraint(
            const trecs::uid_t parent_id,
            const trecs::uid_t child_id,
            const oy::types::constraintTranslation1d_t & constraint
         );

         const std::unordered_set<trecs::uid_t> & getTranslation1dConstraintUids(void) const;

         oy::types::constraintTranslation1d_t & getTranslation1dConstraint(
            const trecs::uid_t entity
         );

         const oy::types::constraintTranslation1d_t & getTranslation1dConstraint(
            const trecs::uid_t entity
         ) const;

         trecs::uid_t addConstantForce(
            const trecs::uid_t child_id,
            const oy::types::forceConstant_t & force
         );

         const std::unordered_set<trecs::uid_t> & getConstantForceUids(void) const;

         oy::types::forceConstant_t & getConstantForce(
            const trecs::uid_t entity
         );

         const oy::types::forceConstant_t & getConstantForce(
            const trecs::uid_t entity
         ) const;

         trecs::uid_t addDragForce(
            const trecs::uid_t child_id,
            const oy::types::forceDrag_t & force
         );

         const std::unordered_set<trecs::uid_t> & getDragForceUids(void) const;

         oy::types::forceDrag_t & getDragForce(
            const trecs::uid_t entity
         );

         const oy::types::forceDrag_t & getDragForce(
            const trecs::uid_t entity
         ) const;

         trecs::uid_t addSpringForce(
            const trecs::uid_t parent_id,
            const trecs::uid_t child_id,
            const oy::types::forceSpring_t & force
         );

         const std::unordered_set<trecs::uid_t> & getSpringForceUids(void) const;

         oy::types::forceSpring_t & getSpringForce(
            const trecs::uid_t entity
         );

         const oy::types::forceSpring_t & getSpringForce(
            const trecs::uid_t entity
         ) const;

         trecs::uid_t addVelocityDamperForce(
            const trecs::uid_t parent_id,
            const trecs::uid_t child_id,
            const oy::types::forceVelocityDamper_t & force
         );

         const std::unordered_set<trecs::uid_t> & getVelocityDamperForceUids(void) const;

         oy::types::forceVelocityDamper_t & getVelocityDamperForce(
            const trecs::uid_t entity
         );

         const oy::types::forceVelocityDamper_t & getVelocityDamperForce(
            const trecs::uid_t entity
         ) const;

         trecs::uid_t addDragTorque(
            const trecs::uid_t child_id,
            const oy::types::torqueDrag_t & force
         );

         const std::unordered_set<trecs::uid_t> & getDragTorqueUids(void) const;

         oy::types::torqueDrag_t & getDragTorque(
            const trecs::uid_t entity
         );

         const oy::types::torqueDrag_t & getDragTorque(
            const trecs::uid_t entity
         ) const;

         void removeEntity(const trecs::uid_t entity);

         // Shoots a ray into the scene and returns the UID of the body whose
         // ray intersection point is closest to the ray's origin.
         // Returns -1 as the bodyId if the ray intersects no bodies.
         // Accepts an optional raycast type.
         // If the raycast filter type is dynamic, then only raycast results
         // against dynamic rigid bodies are reported.
         // If the raycast filter type is stationary, then only raycast results
         // against stationary rigid bodies are reported.
         // If the raycast filter type is all, then raycast results against all
         // rigid body types are reported.
         // The default raycast type is dynamic.
         // The ray start and ray unit vectors should be in global coordinates.
         oy::types::raycastResult_t raycast(
            const Vector3 & ray_start,
            const Vector3 & ray_unit,
            float max_distance,
            oy::types::enumRaycastBodyFilter_t filter_type = oy::types::enumRaycastBodyFilter_t::DYNAMIC
         ) const;

         // Calculates and returns the transform from collider space to world
         // coordinates if 'body_uid' is a valid UID.
         // Returns an identity transformation if 'body_uid' is not valid.
         geometry::types::isometricTransform_t getColliderTransform(
            trecs::uid_t body_uid
         ) const;

         // Calculates and returns the transform from body space to world
         // coordinates if 'body_uid' is a valid UID.
         // Returns an identity transformation if 'body_uid' is not valid.
         geometry::types::transform_t getBodyTransform(trecs::uid_t body_uid) const;

         unsigned int getFrameCount(void);

         float dt(void) const;

         const std::map<int, trecs::uid_t> & getIdMapping(void);

      private:

         trecs::Allocator * allocator_;

         oy::types::scenario_t scenario_;

         // Mapping of scenario rigid body IDs to simulation rigid body IDs.
         std::map<int, trecs::uid_t> scenario_to_sim_id_;

         trecs::query_t dynamic_body_query_;

         trecs::query_t stationary_body_query_;

         trecs::query_t rigid_body_query_;

         trecs::query_t balljoint_constraint_query_;

         trecs::query_t gear_constraint_query_;

         trecs::query_t revolute_joint_constraint_query_;

         trecs::query_t revolute_motor_constraint_query_;

         trecs::query_t rotation_1d_constraint_query_;

         trecs::query_t translation_1d_constraint_query_;

         trecs::query_t constant_force_query_;

         trecs::query_t drag_force_query_;

         trecs::query_t spring_force_query_;

         trecs::query_t velocity_damper_force_query_;

         trecs::query_t drag_torque_query_;

         trecs::query_t edge_query_;

         unsigned int frame_counter_;

         typedef oy::types::cube_t cube_t;
         typedef oy::types::sphere_t sphere_t;
         typedef oy::types::capsule_t capsule_t;
         typedef oy::types::cylinder_t cylinder_t;

         // Systems
         typedef oy::NarrowphaseSystem<cube_t, cylinder_t> cubeCylinderGjk_t;
         typedef oy::NarrowphaseSystem<capsule_t, cylinder_t> capsuleCylinderGjk_t;
         typedef oy::NarrowphaseSystem<cylinder_t, cylinder_t> cylinderCylinderGjk_t;

         typedef oy::CollisionGeometrySystem<cube_t, cylinder_t, geometry::epa::EpaTerminationSmooth> cubeCylinderEpa_t;
         typedef oy::CollisionGeometrySystem<sphere_t, cylinder_t, geometry::epa::EpaTerminationSmooth> sphereCylinderEpa_t;
         typedef oy::CollisionGeometrySystem<capsule_t, cylinder_t, geometry::epa::EpaTerminationSmooth> capsuleCylinderEpa_t;
         typedef oy::CollisionGeometrySystem<cylinder_t, cylinder_t, geometry::epa::EpaTerminationSmooth> cylinderCylinderEpa_t;

         typedef oy::CollisionManifoldSystem<cube_t, cube_t> cubeCubeManifold_t;
         typedef oy::CollisionManifoldSystem<cube_t, sphere_t> cubeSphereManifold_t;
         typedef oy::CollisionManifoldSystem<cube_t, capsule_t> cubeCapsuleManifold_t;
         typedef oy::CollisionManifoldSystem<cube_t, cylinder_t> cubeCylinderManifold_t;
         typedef oy::CollisionManifoldSystem<sphere_t, sphere_t> sphereSphereManifold_t;
         typedef oy::CollisionManifoldSystem<sphere_t, capsule_t> sphereCapsuleManifold_t;
         typedef oy::CollisionManifoldSystem<sphere_t, cylinder_t> sphereCylinderManifold_t;
         typedef oy::CollisionManifoldSystem<capsule_t, capsule_t> capsuleCapsuleManifold_t;
         typedef oy::CollisionManifoldSystem<capsule_t, cylinder_t> capsuleCylinderManifold_t;
         typedef oy::CollisionManifoldSystem<cylinder_t, cylinder_t> cylinderCylinderManifold_t;

         oy::AabbCalculator<cube_t> * cube_aabb_system_;
         oy::AabbCalculator<sphere_t> * sphere_aabb_system_;
         oy::AabbCalculator<capsule_t> * capsule_aabb_system_;
         oy::AabbCalculator<cylinder_t> * cylinder_aabb_system_;

         oy::BroadphaseSystem<cube_t, cube_t> * cube_cube_broadphase_;
         oy::BroadphaseSystem<cube_t, sphere_t> * cube_sphere_broadphase_;
         oy::BroadphaseSystem<cube_t, capsule_t> * cube_capsule_broadphase_;
         oy::BroadphaseSystem<cube_t, cylinder_t> * cube_cylinder_broadphase_;
         oy::BroadphaseSystem<sphere_t, sphere_t> * sphere_sphere_broadphase_;
         oy::BroadphaseSystem<sphere_t, capsule_t> * sphere_capsule_broadphase_;
         oy::BroadphaseSystem<sphere_t, cylinder_t> * sphere_cylinder_broadphase_;
         oy::BroadphaseSystem<capsule_t, capsule_t> * capsule_capsule_broadphase_;
         oy::BroadphaseSystem<capsule_t, cylinder_t> * capsule_cylinder_broadphase_;
         oy::BroadphaseSystem<cylinder_t, cylinder_t> * cylinder_cylinder_broadphase_;

         cubeCylinderGjk_t * cube_cylinder_gjk_;
         capsuleCylinderGjk_t * capsule_cylinder_gjk_;
         cylinderCylinderGjk_t * cylinder_cylinder_gjk_;

         cubeCylinderEpa_t * cube_cylinder_epa_;
         capsuleCylinderEpa_t * capsule_cylinder_epa_;
         cylinderCylinderEpa_t * cylinder_cylinder_epa_;

         oy::SatSystem<cube_t, cube_t> * cube_cube_sat_;
         oy::SatSystem<cube_t, sphere_t> * cube_sphere_sat_;
         oy::SatSystem<cube_t, capsule_t> * cube_capsule_sat_;
         oy::SatSystem<sphere_t, sphere_t> * sphere_sphere_sat_;
         oy::SatSystem<sphere_t, capsule_t> * sphere_capsule_sat_;
         oy::SatSystem<sphere_t, cylinder_t> * sphere_cylinder_sat_;
         oy::SatSystem<capsule_t, capsule_t> * capsule_capsule_sat_;

         ManifoldComponentJankifier * manifold_jankifier_;

         cubeCubeManifold_t * cube_cube_manifold_;
         cubeSphereManifold_t * cube_sphere_manifold_;
         cubeCapsuleManifold_t * cube_capsule_manifold_;
         cubeCylinderManifold_t * cube_cylinder_manifold_;
         sphereSphereManifold_t * sphere_sphere_manifold_;
         sphereCapsuleManifold_t * sphere_capsule_manifold_;
         sphereCylinderManifold_t * sphere_cylinder_manifold_;
         capsuleCapsuleManifold_t * capsule_capsule_manifold_;
         capsuleCylinderManifold_t * capsule_cylinder_manifold_;
         cylinderCylinderManifold_t * cylinder_cylinder_manifold_;

         oy::ConstantForceCalculator * constantForceSystem_;
         oy::VelocityDamperForceCalculator * damperSystem_;
         oy::SpringForceCalculator * springSystem_;
         oy::DragForceCalculator * dragForceSystem_;

         oy::DragTorqueCalculator * dragTorqueSystem_;

         oy::Rk4Integrator * rk4System_;

         oy::GeneralizedForceResetSystem * forque_resetter_system_;
         oy::Rk4Finalizer * rk4_finalizer_system_;
         oy::Rk4MidpointCalculator<0> * rk4_midpoint0_system_;
         oy::Rk4MidpointCalculator<1> * rk4_midpoint1_system_;
         oy::Rk4MidpointCalculator<2> * rk4_midpoint2_system_;
         oy::Rk4MidpointCalculator<3> * rk4_midpoint3_system_;
         oy::Rk4IncrementCalculator<0> * rk4_increment0_system_;
         oy::Rk4IncrementCalculator<1> * rk4_increment1_system_;
         oy::Rk4IncrementCalculator<2> * rk4_increment2_system_;
         oy::Rk4IncrementCalculator<3> * rk4_increment3_system_;

         oy::ConstrainedRigidBodySystem * constrained_body_system_;

         oy::InequalityConstraintCalculatorSystem<oy::types::constraintCollision_t> * collision_calculator_;
         oy::InequalityConstraintCalculatorSystem<oy::types::constraintFriction_t> * friction_calculator_;
         oy::InequalityConstraintCalculatorSystem<oy::types::constraintTorsionalFriction_t> * torsional_friction_calculator_;

         oy::EqualityConstraintCalculatorSystem<oy::types::constraintGear_t> * gear_calculator_;
         oy::EqualityConstraintCalculatorSystem<oy::types::constraintRevoluteMotor_t> * revolute_motor_calculator_;
         oy::EqualityConstraintCalculatorSystem<oy::types::constraintRotation1d_t> * rotation_1d_calculator_;
         oy::EqualityConstraintCalculatorSystem<oy::types::constraintTranslation1d_t> * translation_1d_calculator_;

         oy::ConstraintSolver * constraintSolver_;

         oy::CollisionSystem * collisionSystem_;
         oy::FrictionSystem * frictionSystem_;
         oy::TorsionalFrictionSystem * torsionalFrictionSystem_;

         // Registers the existing systems with the ECS. This just allows the
         // ECS to automatically call some methods in the systems for component
         // registration, query registration, and initialization.
         void registerSystems(void);

         // Registers queries that the Handle uses to retrieve UIDs for bodies,
         // constraints, and forces.
         void registerQueriesAndComponents(void);

         template <typename LinkComponent_T>
         trecs::uid_t addLinkComponent(
            const trecs::uid_t parent_entity,
            const trecs::uid_t child_entity,
            const LinkComponent_T & component,
            const char * component_name
         )
         {
            if (
               (parent_entity == oy::types::null_body_entity) &&
               (child_entity == oy::types::null_body_entity)
            )
            {
               std::cout << "Can't add " << component_name << " between two null body entities\n";
               return oy::types::null_body_entity;
            }

            // Verify that the parent and child entities exist
            const auto rigid_body_entities = allocator_->getQueryEntities(
               rigid_body_query_
            );

            const auto parent_it = std::find(
               rigid_body_entities.begin(),
               rigid_body_entities.end(),
               parent_entity
            );

            const auto child_it = std::find(
               rigid_body_entities.begin(),
               rigid_body_entities.end(),
               child_entity
            );

            if (
               (parent_entity != oy::types::null_body_entity) &&
               (parent_it == rigid_body_entities.end())
            )
            {
               std::cout << "Parent entity for " << component_name << " not found in rigid bodies: " << parent_entity << "\n";
               return oy::types::null_body_entity;
            }

            if (
               (child_entity != oy::types::null_body_entity) &&
               (child_it == rigid_body_entities.end())
            )
            {
               std::cout << "Child entity for " << component_name << " not found in rigid bodies: " << child_entity << "\n";
               return oy::types::null_body_entity;
            }

            const trecs::uid_t entity = allocator_->addEntity(
               parent_entity, child_entity
            );

            if(!allocator_->addComponent(entity, component))
            {
               std::cout << "Couldn't add " << component_name << " with UID " << entity << "\n";
               allocator_->removeEntity(entity);
               return oy::types::null_body_entity;
            }

            return entity;
         }

         template <typename LinkComponent_T>
         trecs::uid_t addCompositeLinkComponent(
            const trecs::uid_t parent_entity,
            const trecs::uid_t child_entity,
            const LinkComponent_T & component,
            const char * component_name
         )
         {
            const trecs::uid_t entity = addLinkComponent<LinkComponent_T>(
               parent_entity, child_entity, component, component_name
            );

            oy::types::compositeConstraintJoint_t temp_composite = {
               0, {-1, -1, -1}, 0, {-1, -1, -1}
            };

            if (!allocator_->addComponent(entity, temp_composite))
            {
               std::cout << "Couldn't add composite component to " << component_name << " with UID " << entity << "\n";
               allocator_->removeEntity(entity);
               return oy::types::null_body_entity;
            }

            return entity;
         }

         template <typename LinkComponent_T>
         LinkComponent_T & getLinkComponent(
            const trecs::uid_t entity, const std::string & component_name
         )
         {
            LinkComponent_T * constraint = allocator_->getComponent<LinkComponent_T>(entity);

            if (constraint == nullptr)
            {
               std::string error_string = "Couldn't find " + component_name + " at entity ";
               error_string += std::to_string(entity);

               throw std::invalid_argument(error_string);
            }

            return *constraint;
         }

         template <typename LinkComponent_T>
         const LinkComponent_T & getLinkComponent(
            const trecs::uid_t entity, const std::string & component_name
         ) const
         {
            LinkComponent_T * component = allocator_->getComponent<LinkComponent_T>(entity);

            if (component == nullptr)
            {
               std::string error_string = "Couldn't find " + component_name + " at entity ";
               error_string += std::to_string(entity);

               throw std::invalid_argument(error_string);
            }

            return *component;
         }

         template <typename LinkComponent_T>
         void removeCompositeLinkComponent(const trecs::uid_t entity)
         {
            if (!allocator_->hasComponent<LinkComponent_T>(entity))
            {
               return;
            }

            oy::types::compositeConstraintJoint_t * composite_joint = \
               allocator_->getComponent<oy::types::compositeConstraintJoint_t>(
                  entity
               );

            for (unsigned int i = 0; i < composite_joint->numRotationConstraints; ++i)
            {
               allocator_->removeEntity(composite_joint->rotationConstraintIds[i]);
            }

            for (unsigned int i = 0; i < composite_joint->numTranslationConstraints; ++i)
            {
               allocator_->removeEntity(composite_joint->translationConstraintIds[i]);
            }

            allocator_->removeEntity(entity);
         }
   };

}

#endif
