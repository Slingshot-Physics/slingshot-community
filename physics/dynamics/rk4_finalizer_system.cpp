#include "rk4_finalizer_system.hpp"

#include "slingshot_types.hpp"

#include "rk4_integrator.hpp"

namespace oy
{
   void Rk4Finalizer::registerComponents(trecs::Allocator & allocator) const
   {
      allocator.registerComponent<oy::types::rigidBody_t>();
      allocator.registerComponent<oy::types::DynamicBody>();
      allocator.registerComponent<oy::types::rk4Increment_t<0> >();
      allocator.registerComponent<oy::types::rk4Increment_t<1> >();
      allocator.registerComponent<oy::types::rk4Increment_t<2> >();
      allocator.registerComponent<oy::types::rk4Increment_t<3> >();
   }

   void Rk4Finalizer::registerQueries(trecs::Allocator & allocator)
   {
      dynamic_body_query_ = allocator.addArchetypeQuery<
         oy::types::rigidBody_t,
         oy::types::DynamicBody,
         oy::types::rk4Increment_t<0>,
         oy::types::rk4Increment_t<1>,
         oy::types::rk4Increment_t<2>,
         oy::types::rk4Increment_t<3>
      >();
   }

   void Rk4Finalizer::update(trecs::Allocator & allocator) const
   {
      auto rigid_bodies = allocator.getComponents<oy::types::rigidBody_t>();
      
      const auto rk4_increment0s = allocator.getComponents<oy::types::rk4Increment_t<0> >();
      const auto rk4_increment1s = allocator.getComponents<oy::types::rk4Increment_t<1> >();
      const auto rk4_increment2s = allocator.getComponents<oy::types::rk4Increment_t<2> >();
      const auto rk4_increment3s = allocator.getComponents<oy::types::rk4Increment_t<3> >();

      const auto & dynamic_body_entities = allocator.getQueryEntities(dynamic_body_query_);

      const float dt = oy::integrator::dt_;

      for (const auto entity : dynamic_body_entities)
      {
         auto body = rigid_bodies[entity];

         const auto lin_pos_dot_k0 = rk4_increment0s[entity]->linPosDot;
         const auto lin_pos_dot_k1 = rk4_increment1s[entity]->linPosDot;
         const auto lin_pos_dot_k2 = rk4_increment2s[entity]->linPosDot;
         const auto lin_pos_dot_k3 = rk4_increment3s[entity]->linPosDot;

         body->linPos += (dt / 6.f) * (
            lin_pos_dot_k0 + \
            2.f * lin_pos_dot_k1 + \
            2.f * lin_pos_dot_k2 + \
            lin_pos_dot_k3
         );

         const auto ql2b_dot_k0 = rk4_increment0s[entity]->ql2bDot;
         const auto ql2b_dot_k1 = rk4_increment1s[entity]->ql2bDot;
         const auto ql2b_dot_k2 = rk4_increment2s[entity]->ql2bDot;
         const auto ql2b_dot_k3 = rk4_increment3s[entity]->ql2bDot;

         body->ql2b += (dt / 6.f) * (
            ql2b_dot_k0 + \
            2.f * ql2b_dot_k1 + \
            2.f * ql2b_dot_k2 + \
            ql2b_dot_k3
         );

         const auto lin_vel_dot_k0 = rk4_increment0s[entity]->linVelDot;
         const auto lin_vel_dot_k1 = rk4_increment1s[entity]->linVelDot;
         const auto lin_vel_dot_k2 = rk4_increment2s[entity]->linVelDot;
         const auto lin_vel_dot_k3 = rk4_increment3s[entity]->linVelDot;

         body->linVel += (dt / 6.f) * (
            lin_vel_dot_k0 + \
            2.f * lin_vel_dot_k1 + \
            2.f * lin_vel_dot_k2 + \
            lin_vel_dot_k3
         );

         const auto ang_vel_dot_k0 = rk4_increment0s[entity]->angVelDot;
         const auto ang_vel_dot_k1 = rk4_increment1s[entity]->angVelDot;
         const auto ang_vel_dot_k2 = rk4_increment2s[entity]->angVelDot;
         const auto ang_vel_dot_k3 = rk4_increment3s[entity]->angVelDot;

         body->angVel += (dt / 6.f) * (
            ang_vel_dot_k0 + \
            2.f * ang_vel_dot_k1 + \
            2.f * ang_vel_dot_k2 + \
            ang_vel_dot_k3
         );
      }
   }
}
