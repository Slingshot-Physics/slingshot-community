#ifndef INEQUALITY_CONSTRAINT_CALCULATOR_SYSTEM_HEADER
#define INEQUALITY_CONSTRAINT_CALCULATOR_SYSTEM_HEADER

#include "allocator.hpp"

#include "system.hpp"

#include "constraint_output.hpp"
#include "slingshot_types.hpp"
#include "rk4_integrator.hpp"

#include <iostream>
#include <vector>

namespace oy
{
   // Calculates the constraint output for a particular type of constraint.
   template <typename Constraint_T>
   class InequalityConstraintCalculatorSystem : public trecs::System
   {
      private:

         typedef oy::types::categoryInequalityConstraint_t<Constraint_T> InequalityConstraintTag_T;

         typedef std::vector<trecs::edge_t> edges_t;

      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            // The inequality constraint tag keeps the collision constraint
            // output separate from the friction constraint output.
            allocator.registerComponent<InequalityConstraintTag_T>();
            allocator.registerComponent<std::vector<Constraint_T> >();
            allocator.registerComponent<std::vector<oy::types::constraintOutput_t> >();
            allocator.registerComponent<edges_t>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            rigid_body_query_ = allocator.addArchetypeQuery<oy::types::rigidBody_t>();

            constraints_query_ = allocator.addArchetypeQuery<std::vector<Constraint_T> >();

            inequality_constraint_outputs_query_ = allocator.addArchetypeQuery<
               InequalityConstraintTag_T,
               std::vector<oy::types::constraintOutput_t>,
               edges_t
            >();
         }

         void initialize(trecs::Allocator & allocator) override
         {
            trecs::uid_t new_constraint_output_entity = allocator.addEntity();

            std::vector<oy::types::constraintOutput_t> temp_constraint_output;
            if (!allocator.addComponent(new_constraint_output_entity, temp_constraint_output))
            {
               std::cout << "Inequality constraint output system couldn't add a vector of constraint output\n";
            }

            edges_t temp_edges;
            if (!allocator.addComponent(new_constraint_output_entity, temp_edges))
            {
               std::cout << "Inequality constraint output system couldn't add a vector of trecs::edges\n";
            }

            InequalityConstraintTag_T temp_inequality_constraing_tag;
            if (!allocator.addComponent(new_constraint_output_entity, temp_inequality_constraing_tag))
            {
               std::cout << "Inequality constraint output system couldn't add an inequality constraint tag\n";
            }
         }

         void update(trecs::Allocator & allocator) const
         {
            const auto & constraint_entities = allocator.getQueryEntities(
               constraints_query_
            );

            if (constraint_entities.size() == 0)
            {
               std::cout << "Inequality constraint calculator couldn't find an entity with a vector of constraints on it\n";
               return;
            }

            const std::vector<Constraint_T> * constraints = \
               allocator.getComponent<std::vector<Constraint_T> >(
                  *constraint_entities.begin()
               );

            const auto & inequality_constraint_output_entities = allocator.getQueryEntities(
               inequality_constraint_outputs_query_
            );

            if (inequality_constraint_output_entities.size() == 0)
            {
               std::cout << "Inequality constraint calculator couldn't find an entity with an inequality constraint tag and constraint output vector on it\n";
               return;
            }

            std::vector<oy::types::constraintOutput_t> * constraint_outputs = \
               allocator.getComponent<std::vector<oy::types::constraintOutput_t> >(
                  *inequality_constraint_output_entities.begin()
               );

            edges_t * constraint_edges = \
               allocator.getComponent<edges_t>(
                  *inequality_constraint_output_entities.begin()
               );

            constraint_outputs->clear();
            constraint_edges->clear();

            const auto bodies = allocator.getComponents<oy::types::rigidBody_t>();

            float dt = oy::integrator::dt_;

            for (const auto & constraint : *constraints)
            {
               bool body_a_stationary = \
                  allocator.hasComponent<oy::types::StationaryBody>(
                     constraint.bodyIdA
                  );
               bool body_b_stationary = \
                  allocator.hasComponent<oy::types::StationaryBody>(
                     constraint.bodyIdB
                  );

               bool body_a_active = constraint.bodyIdA != oy::types::null_body_entity;
               bool body_b_active = constraint.bodyIdB != oy::types::null_body_entity;

               const oy::types::rigidBody_t & body_a = (
                  body_a_active ? *bodies[constraint.bodyIdA] : *bodies[constraint.bodyIdB]
               );
               const oy::types::rigidBody_t & body_b = (
                  body_b_active ? *bodies[constraint.bodyIdB] : *bodies[constraint.bodyIdA]
               );

               oy::types::constraintOutput_t constraint_output = \
                  oy::calculateConstraintOutput(
                     body_a_stationary,
                     body_b_stationary,
                     constraint.bodyIdA,
                     constraint.bodyIdB,
                     body_a,
                     body_b,
                     constraint,
                     dt
                  );

               constraint_outputs->push_back(constraint_output);

               trecs::edge_t constraint_edge;
               constraint_edge.nodeIdA = constraint.bodyIdA;
               constraint_edge.nodeIdB = constraint.bodyIdB;
               constraint_edge.flag = trecs::TRANSITIVE;

               if (!body_a_active || body_a_stationary)
               {
                  constraint_edge.flag = trecs::NODE_A_TERMINAL;
               }
               else if (!body_b_active || body_b_stationary)
               {
                  constraint_edge.flag = trecs::NODE_B_TERMINAL;
               }

               constraint_edges->push_back(constraint_edge);
            }
         }

      private:
         trecs::query_t rigid_body_query_;

         trecs::query_t constraints_query_;

         trecs::query_t inequality_constraint_outputs_query_;
   };
}

#endif
