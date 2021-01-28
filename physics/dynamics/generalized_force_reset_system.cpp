#include "generalized_force_reset_system.hpp"

#include "slingshot_types.hpp"

namespace oy
{
   void GeneralizedForceResetSystem::registerComponents(trecs::Allocator & allocator) const
   {
      allocator.registerComponent<oy::types::generalizedForce_t>();
   }

   void GeneralizedForceResetSystem::registerQueries(trecs::Allocator & allocator)
   {
      generalized_force_query_ = allocator.addArchetypeQuery<oy::types::generalizedForce_t>();
   }

   void GeneralizedForceResetSystem::update(trecs::Allocator & allocator) const
   {
      auto generalized_forces = allocator.getComponents<oy::types::generalizedForce_t>();

      const auto & entities = allocator.getQueryEntities(generalized_force_query_);

      for (const auto entity : entities)
      {
         oy::types::generalizedForce_t * g_force = generalized_forces[entity];

         g_force->appliedForce.Initialize(0.f, 0.f, 0.f);
         g_force->appliedTorque.Initialize(0.f, 0.f, 0.f);
      }
   }
}

