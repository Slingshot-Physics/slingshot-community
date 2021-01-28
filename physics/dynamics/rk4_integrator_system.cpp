#include "rk4_integrator_system.hpp"

#include "slingshot_types.hpp"
#include "rk4_integrator.hpp"

namespace oy
{
   void Rk4Integrator::registerComponents(trecs::Allocator & allocator) const
   {
      allocator.registerComponent<oy::types::rigidBody_t>();
      allocator.registerComponent<oy::types::generalizedForce_t>();
      allocator.registerComponent<oy::types::DynamicBody>();
      allocator.registerComponent<oy::types::StationaryBody>();
   }

   void Rk4Integrator::registerQueries(trecs::Allocator & allocator)
   {
      dynamic_body_query_ = allocator.addArchetypeQuery<
         oy::types::rigidBody_t,
         oy::types::DynamicBody,
         oy::types::generalizedForce_t
      >();
   }

   void Rk4Integrator::update(trecs::Allocator & allocator) const
   {
      auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
      auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

      const auto & entities = allocator.getQueryEntities(dynamic_body_query_);

      for (const auto entity : entities)
      {
         oy::types::rigidBody_t * body = bodies[entity];
         oy::types::generalizedForce_t * forque = forques[entity];

         oy::integrator::forwardStep(*body, *forque);

         if (body->linPos.hasNan())
         {
            std::cout << "Found nan in rigid body with entity UID: " << entity << "\n";
         }

         forque->appliedForce.Initialize(0, 0, 0);
         forque->appliedTorque.Initialize(0, 0, 0);
      }
   }
}
