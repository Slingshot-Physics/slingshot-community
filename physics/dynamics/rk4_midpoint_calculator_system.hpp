#ifndef RK4_MIDPOINT_CALCULATOR_SYSTEM_HEADER
#define RK4_MIDPOINT_CALCULATOR_SYSTEM_HEADER

#include "allocator.hpp"
#include "system.hpp"

#include "rk4_integrator.hpp"
#include "slingshot_types.hpp"

#include <type_traits>

namespace oy
{
   template <int N>
   class Rk4MidpointCalculator : public trecs::System
   {
      public:
         Rk4MidpointCalculator(void)
         {
            static_assert(N >= 0, "RK4 stage must be >= 0");
            static_assert(N <= 3, "RK4 stage must be <= 3");
         }

         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<oy::types::rigidBody_t>();
            allocator.registerComponent<oy::types::DynamicBody>();
            allocator.registerComponent<oy::types::rk4Midpoint_t>();
            if (N > 0)
            {
               allocator.registerComponent<oy::types::rk4Increment_t<N - 1> >();
            }
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            if (N > 0)
            {
               integrated_body_query_ = allocator.addArchetypeQuery<
                  oy::types::rigidBody_t,
                  oy::types::rk4Increment_t<N - 1>,
                  oy::types::rk4Midpoint_t
               >();
            }
            else
            {
               integrated_body_query_ = allocator.addArchetypeQuery<
                  oy::types::rigidBody_t,
                  oy::types::rk4Midpoint_t
               >();
            }
         }

         template <int Dummy = N>
         auto update(trecs::Allocator & allocator) const ->
            typename std::enable_if< (Dummy > 0), void >::type
         {
            auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
            auto rk4_increments = allocator.getComponents<oy::types::rk4Increment_t<N - 1> >();
            auto rk4_midpoints = allocator.getComponents<oy::types::rk4Midpoint_t>();

            const auto & entities = allocator.getQueryEntities(integrated_body_query_);

            const float dt = dts_[N];

            for (const auto entity : entities)
            {
               const auto body_state = bodies[entity];
               const auto body_state_derivs = rk4_increments[entity];

               auto rk4_midpoint = rk4_midpoints[entity];

               rk4_midpoint->mass = body_state->mass;
               rk4_midpoint->inertiaTensor = body_state->inertiaTensor;
               rk4_midpoint->linPos = body_state->linPos + dt * body_state_derivs->linPosDot;
               rk4_midpoint->linVel = body_state->linVel + dt * body_state_derivs->linVelDot;
               rk4_midpoint->ql2b = body_state->ql2b + dt * body_state_derivs->ql2bDot;
               rk4_midpoint->angVel = body_state->angVel + dt * body_state_derivs->angVelDot;
            }
         }

         template <int Dummy = N>
         auto update(trecs::Allocator & allocator) const ->
            typename std::enable_if< Dummy == 0, void >::type
         {
            auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
            auto rk4_midpoints = allocator.getComponents<oy::types::rk4Midpoint_t>();

            const auto & entities = allocator.getQueryEntities(integrated_body_query_);

            for (const auto entity : entities)
            {
               const auto body_state = bodies[entity];

               auto rk4_midpoint = rk4_midpoints[entity];

               rk4_midpoint->linPos = body_state->linPos;
               rk4_midpoint->linVel = body_state->linVel;
               rk4_midpoint->ql2b = body_state->ql2b;
               rk4_midpoint->angVel = body_state->angVel;
               rk4_midpoint->mass = body_state->mass;
               rk4_midpoint->inertiaTensor = body_state->inertiaTensor;
            }
         }

      private:
         trecs::query_t integrated_body_query_;

         const float dts_[4] = {
            0.f,
            oy::integrator::dt_ / 2.f,
            oy::integrator::dt_ / 2.f,
            oy::integrator::dt_
         };

   };
}

#endif
