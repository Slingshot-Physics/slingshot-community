#ifndef EQUALITY_CONSTRAINT_CALCULATOR_SYSTEM_HEADER
#define EQUALITY_CONSTRAINT_CALCULATOR_SYSTEM_HEADER

#include "allocator.hpp"

#include "system.hpp"

#include "constraint_output.hpp"
#include "slingshot_types.hpp"
#include "rk4_integrator.hpp"

namespace oy
{
   // Calculates the constraint output for a particular type of constraint.
   template <typename Constraint_T>
   class EqualityConstraintCalculatorSystem : public trecs::System
   {
      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<Constraint_T>();
            allocator.registerComponent<oy::types::constraintOutput_t>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            rigid_body_query_ = allocator.addArchetypeQuery<oy::types::rigidBody_t>();

            constraint_query_ = allocator.addArchetypeQuery<Constraint_T>();
         }

         void update(trecs::Allocator & allocator) const
         {
            const auto constraint_entities = allocator.getQueryEntities(constraint_query_);

            const auto constraints = allocator.getComponents<Constraint_T>();
            const auto bodies = allocator.getComponents<oy::types::rigidBody_t>();

            float dt = oy::integrator::dt_;

            for (const auto entity : constraint_entities)
            {
               trecs::edge_t constraint_edge = allocator.getEdge(entity);

               bool body_a_stationary = \
                  allocator.hasComponent<oy::types::StationaryBody>(
                     constraint_edge.nodeIdA
                  );
               bool body_b_stationary = \
                  allocator.hasComponent<oy::types::StationaryBody>(
                     constraint_edge.nodeIdB
                  );

               bool body_a_active = (
                  (constraint_edge.nodeIdA != oy::types::null_body_entity)
               );
               bool body_b_active = (
                  (constraint_edge.nodeIdB != oy::types::null_body_entity)
               );

               const oy::types::rigidBody_t & body_a = (
                  body_a_active ? *bodies[constraint_edge.nodeIdA] : *bodies[constraint_edge.nodeIdB]
               );
               const oy::types::rigidBody_t & body_b = (
                  body_b_active ? *bodies[constraint_edge.nodeIdB] : *bodies[constraint_edge.nodeIdA]
               );

               oy::types::constraintOutput_t constraint_output = \
                  oy::calculateConstraintOutput(
                     body_a_stationary,
                     body_b_stationary,
                     constraint_edge.nodeIdA,
                     constraint_edge.nodeIdB,
                     body_a,
                     body_b,
                     *constraints[entity],
                     dt
                  );

               allocator.updateComponent(entity, constraint_output);
            }
         }

      private:
         trecs::query_t rigid_body_query_;

         trecs::query_t constraint_query_;
   };
}

#endif
