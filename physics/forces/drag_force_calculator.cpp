#include "drag_force_calculator.hpp"

#include "drag_force.hpp"

namespace oy
{
   void DragForceCalculator::registerComponents(trecs::Allocator & allocator) const
   {
      allocator.registerComponent<oy::types::rigidBody_t>();
      allocator.registerComponent<oy::types::generalizedForce_t>();
      allocator.registerComponent<oy::types::forceDrag_t>();
   }

   void DragForceCalculator::registerQueries(trecs::Allocator & allocator)
   {
      drag_force_query_ = allocator.addArchetypeQuery<oy::types::forceDrag_t>();
   }

   void DragForceCalculator::update(trecs::Allocator & allocator) const
   {
      const auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
      const auto force_configs = allocator.getComponents<oy::types::forceDrag_t>();
      const auto edges = allocator.getComponents<trecs::edge_t>();
      auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

      const auto & force_config_entities = allocator.getQueryEntities(drag_force_query_);

      for (const auto entity : force_config_entities)
      {
         const trecs::edge_t & edge = *edges[entity];
         const oy::types::rigidBody_t & body = *bodies[edge.nodeIdB];
         const oy::types::forceDrag_t & force_config = *force_configs[entity];
         oy::types::generalizedForce_t & forque = *forques[edge.nodeIdB];

         oy::forces::drag::evaluateForce(
            edge.nodeIdA,
            edge.nodeIdB,
            force_config,
            body,
            body,
            forque,
            forque
         );
      }
   }
}
