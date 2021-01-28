#include "constant_force_calculator.hpp"

#include "constant_force.hpp"

namespace oy
{
   void ConstantForceCalculator::registerComponents(trecs::Allocator & allocator) const
   {
      allocator.registerComponent<oy::types::rigidBody_t>();
      allocator.registerComponent<oy::types::generalizedForce_t>();
      allocator.registerComponent<oy::types::forceConstant_t>();
   }

   void ConstantForceCalculator::registerQueries(trecs::Allocator & allocator)
   {
      constant_force_query_ = allocator.addArchetypeQuery<oy::types::forceConstant_t>();
   }

   void ConstantForceCalculator::update(trecs::Allocator & allocator) const
   {
      const auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
      const auto constant_force_configs = allocator.getComponents<oy::types::forceConstant_t>();
      const auto edges = allocator.getComponents<trecs::edge_t>();
      auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

      const auto & constant_force_entities = allocator.getQueryEntities(constant_force_query_);

      for (const auto entity : constant_force_entities)
      {
         const trecs::edge_t & edge = *edges[entity];
         const oy::types::rigidBody_t & body = *bodies[edge.nodeIdB];
         const oy::types::forceConstant_t & force_config = *constant_force_configs[entity];
         oy::types::generalizedForce_t & forque = *forques[edge.nodeIdB];

         oy::forces::constant::evaluateForce(
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
