#include "constrained_rigidbody_system.hpp"

#include "slingshot_types.hpp"
#include "rk4_integrator.hpp"

namespace oy
{
   void ConstrainedRigidBodySystem::registerComponents(
      trecs::Allocator & allocator
   ) const
   {
      allocator.registerComponent<oy::types::constrainedRigidBody_t>();
      allocator.registerComponent<oy::types::DynamicBody>();
      allocator.registerComponent<oy::types::StationaryBody>();
   }

   void ConstrainedRigidBodySystem::registerQueries(
      trecs::Allocator & allocator
   )
   {
      dynamic_body_query_ = allocator.addArchetypeQuery<oy::types::rigidBody_t, oy::types::DynamicBody>();
   }

   void ConstrainedRigidBodySystem::update(
      trecs::Allocator & allocator
   ) const
   {
      const auto dynamic_body_entities = allocator.getQueryEntities(dynamic_body_query_);
      const auto rigid_bodies = allocator.getComponents<oy::types::rigidBody_t>();
      const auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

      const float dt = oy::integrator::dt_;

      for (const auto entity : dynamic_body_entities)
      {
         const oy::types::rigidBody_t * rigid_body = rigid_bodies[entity];
         const oy::types::generalizedForce_t * forque = forques[entity];

         const Matrix33 R_B_to_W = rigid_body->ql2b.rotationMatrix().transpose();

         const Vector3 ang_vel_dot_B = (
            (~rigid_body->inertiaTensor) * (
               forque->appliedTorque - rigid_body->angVel.crossProduct(
                  rigid_body->inertiaTensor * rigid_body->angVel
               )
            )
         );

         oy::types::constrainedRigidBody_t constrained_body;
         constrained_body.invMass = 1.f / rigid_body->mass;
         constrained_body.invInertia = R_B_to_W * (~rigid_body->inertiaTensor) * R_B_to_W.transpose();

         constrained_body.nextQVel.assignSlice(
            0,
            0,
            rigid_body->linVel + (dt / rigid_body->mass) * forque->appliedForce
         );
         constrained_body.nextQVel.assignSlice(
            3,
            0,
            R_B_to_W * (
               rigid_body->angVel + dt * ang_vel_dot_B
            )
         );

         allocator.updateComponent(entity, constrained_body);
      }
   }
}
