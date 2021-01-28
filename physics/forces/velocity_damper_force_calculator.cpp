#include "velocity_damper_force_calculator.hpp"

#include "velocity_damper_force.hpp"

namespace oy
{
   void VelocityDamperForceCalculator::registerComponents(trecs::Allocator & allocator) const
   {
      allocator.registerComponent<oy::types::rigidBody_t>();
      allocator.registerComponent<oy::types::generalizedForce_t>();
      allocator.registerComponent<oy::types::forceVelocityDamper_t>();
   }

   void VelocityDamperForceCalculator::registerQueries(trecs::Allocator & allocator)
   {
      damper_query_ = allocator.addArchetypeQuery<oy::types::forceVelocityDamper_t>();
   }

   void VelocityDamperForceCalculator::update(trecs::Allocator & allocator) const
   {
      const auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
      const auto dampers = allocator.getComponents<oy::types::forceVelocityDamper_t>();
      const auto edges = allocator.getComponents<trecs::edge_t>();
      auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

      const auto & edge_entities = allocator.getQueryEntities(damper_query_);

      for (const auto entity : edge_entities)
      {
         const trecs::edge_t & edge = *edges[entity];
         if (edge.edgeId == -1)
         {
            continue;
         }

         const oy::types::rigidBody_t & parent_body = (edge.nodeIdA != -1) ? *bodies[edge.nodeIdA] : *bodies[edge.nodeIdB];
         const oy::types::rigidBody_t & child_body = (edge.nodeIdB != -1) ? *bodies[edge.nodeIdB] : *bodies[edge.nodeIdA];
         oy::types::generalizedForce_t & parent_forque = (edge.nodeIdA != oy::types::null_body_entity) ? *forques[edge.nodeIdA] : *forques[edge.nodeIdB];
         oy::types::generalizedForce_t & child_forque = (edge.nodeIdB != oy::types::null_body_entity) ? *forques[edge.nodeIdB] : *forques[edge.nodeIdA];

         const oy::types::forceVelocityDamper_t * damper = dampers[entity];

         if (damper == nullptr)
         {
            continue;
         }

         oy::forces::velocity_damper::evaluateForce(
            edge.nodeIdA,
            edge.nodeIdB,
            *damper,
            parent_body,
            child_body,
            parent_forque,
            child_forque
         );
      }
   }
}
