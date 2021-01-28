#include "spring_force_calculator.hpp"

#include "spring_force.hpp"

namespace oy
{
   void SpringForceCalculator::registerComponents(trecs::Allocator & allocator) const
   {
      allocator.registerComponent<oy::types::rigidBody_t>();
      allocator.registerComponent<oy::types::generalizedForce_t>();
      allocator.registerComponent<oy::types::forceSpring_t>();
   }

   void SpringForceCalculator::registerQueries(trecs::Allocator & allocator)
   {
      spring_query_ = allocator.addArchetypeQuery<oy::types::forceSpring_t>();
   }

   void SpringForceCalculator::update(trecs::Allocator & allocator) const
   {
      const auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
      const auto springs = allocator.getComponents<oy::types::forceSpring_t>();
      const auto edges = allocator.getComponents<trecs::edge_t>();
      auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

      const auto & edge_entities = allocator.getQueryEntities(spring_query_);

      for (const auto entity : edge_entities)
      {
         const trecs::edge_t & edge = *edges[entity];
         if (edge.edgeId == -1)
         {
            std::cout << "bad edge id: " << entity << "\n";
            continue;
         }

         const oy::types::rigidBody_t & parent_body = (edge.nodeIdA != oy::types::null_body_entity) ? *bodies[edge.nodeIdA] : *bodies[edge.nodeIdB];
         const oy::types::rigidBody_t & child_body = (edge.nodeIdB != oy::types::null_body_entity) ? *bodies[edge.nodeIdB] : *bodies[edge.nodeIdA];

         const oy::types::forceSpring_t * spring = springs[entity];
         oy::types::generalizedForce_t & parent_forque = (edge.nodeIdA != oy::types::null_body_entity) ? *forques[edge.nodeIdA] : *forques[edge.nodeIdB];
         oy::types::generalizedForce_t & child_forque = (edge.nodeIdB != oy::types::null_body_entity) ? *forques[edge.nodeIdB] : *forques[edge.nodeIdA];

         if (spring == nullptr)
         {
            std::cout << "null spring at entity id: " << (entity & 0xfffff) << "\n";
            continue;
         }

         oy::forces::spring::evaluateForce(
            edge.nodeIdA,
            edge.nodeIdB,
            *spring,
            parent_body,
            child_body,
            parent_forque,
            child_forque
         );
      }
   }
}
