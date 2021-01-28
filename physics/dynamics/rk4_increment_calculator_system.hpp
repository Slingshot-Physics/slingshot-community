#ifndef RK4_INCREMENT_CALCULATOR_SYSTEM_HEADER
#define RK4_INCREMENT_CALCULATOR_SYSTEM_HEADER

#include "allocator.hpp"
#include "system.hpp"

#include "rk4_integrator.hpp"
#include "slingshot_types.hpp"

#include <type_traits>
namespace oy
{
   // The RK4 increment calculator only needs a generalized force, an RK4
   // midpoint, and an RK4 increment for the given increment number. It only
   // really performs half of the calculations of the RK4 increment - the
   // position derivative and quaternion derivative. The other two parts of
   // the RK4 increment come from forces.
   template <int N>
   class Rk4IncrementCalculator : public trecs::System
   {
      public:
         Rk4IncrementCalculator(void)
         {
            static_assert(N >= 0, "RK4 stage must be >= 0");
            static_assert(N <= 3, "RK4 stage must be <= 3");
         }

         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<oy::types::rk4Increment_t<N> >();
            allocator.registerComponent<oy::types::rk4Midpoint_t>();
            allocator.registerComponent<oy::types::generalizedForce_t>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            rk4_body_query_ = allocator.addArchetypeQuery<
               oy::types::rk4Increment_t<N>,
               oy::types::rk4Midpoint_t,
               oy::types::generalizedForce_t
            >();
         }

         void update(trecs::Allocator & allocator) const
         {
            const auto & rk4_body_entities = allocator.getQueryEntities(rk4_body_query_);

            const auto rk4_midpoints = allocator.getComponents<oy::types::rk4Midpoint_t>();
            const auto forques = allocator.getComponents<oy::types::generalizedForce_t>();
            auto rk4_increments = allocator.getComponents<oy::types::rk4Increment_t<N> >();

            for (const auto entity : rk4_body_entities)
            {
               const auto rk4_midpoint = rk4_midpoints[entity];
               const auto forque = forques[entity];
               auto rk4_increment = rk4_increments[entity];

               rk4_increment->linPosDot = rk4_midpoint->linVel;
               rk4_increment->linVelDot = oy::integrator::linVelDot(
                  forque->appliedForce, rk4_midpoint->mass
               );
               rk4_increment->ql2bDot = oy::integrator::attitudeDot(
                  rk4_midpoint->ql2b, rk4_midpoint->angVel
               );
               rk4_increment->angVelDot = oy::integrator::angVelDot(
                  forque->appliedTorque,
                  rk4_midpoint->angVel,
                  rk4_midpoint->inertiaTensor
               );
            }
         }

      private:
         trecs::query_t rk4_body_query_;
   };
}

#endif
