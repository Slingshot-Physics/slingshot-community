#ifndef MANIFOLD_COMPONENT_JANKIFIER
#define MANIFOLD_COMPONENT_JANKIFIER

#include "allocator.hpp"

#include "system.hpp"

#include "slingshot_types.hpp"

#include <vector>

namespace oy
{
   class ManifoldComponentJankifier : public trecs::System
   {
      typedef std::vector<oy::types::collisionContactManifold_t> collisionManifolds_t;

      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<collisionManifolds_t>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            contact_manifold_query_ = allocator.addArchetypeQuery<
               collisionManifolds_t
            >();
         }

         void update(trecs::Allocator & allocator) const
         {
            const auto & manifold_entities = allocator.getQueryEntities(contact_manifold_query_);

            if (manifold_entities.size() == 0)
            {
               std::cout << "Manifold jankifier didn't find any entities that have a vector of contact manifolds attached to them\n";
               return;
            }

            auto manifolds = allocator.getComponent<std::vector<oy::types::collisionContactManifold_t> >(
               *manifold_entities.begin()
            );

            manifolds->clear();
         }

      private:
         trecs::query_t contact_manifold_query_;

   };
}

#endif
