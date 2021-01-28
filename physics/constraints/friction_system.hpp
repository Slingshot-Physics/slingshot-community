#ifndef FRICTION_CONSTRAINT_SYSTEM_HEADER
#define FRICTION_CONSTRAINT_SYSTEM_HEADER

#include "allocator.hpp"

#include "slingshot_types.hpp"
#include "system.hpp"

#include <vector>

namespace oy
{
   class FrictionSystem : public trecs::System
   {
      private:
         typedef std::vector<oy::types::constraintFriction_t> constraintFrictions_t;
         typedef std::vector<oy::types::collisionContactManifold_t> collisionManifolds_t;

      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<constraintFrictions_t>();
            allocator.registerComponent<collisionManifolds_t>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            manifold_query_ = allocator.addArchetypeQuery<
               collisionManifolds_t
            >();

            friction_query_ = allocator.addArchetypeQuery<
               constraintFrictions_t
            >();
         }

         void initialize(trecs::Allocator & allocator) override
         {
            // The entity that will hold the vector of friction constraints.
            trecs::uid_t constraint_frictions_entity = allocator.addEntity();

            constraintFrictions_t temp_frictions;
            if (!allocator.addComponent(constraint_frictions_entity, temp_frictions))
            {
               std::cout << "Error adding vector of friction constraints from the collision friction system\n";
            }
         }

         void update(trecs::Allocator & allocator) const
         {
            const auto manifold_entities = allocator.getQueryEntities(
               manifold_query_
            );

            if (manifold_entities.size() == 0)
            {
               std::cout << "Friction system didn't find any entities that have a vector of contact manifolds attached to them\n";
               return;
            }

            const auto & manifolds = *allocator.getComponent<collisionManifolds_t>(
               *manifold_entities.begin()
            );

            const auto frictions_entities = allocator.getQueryEntities(
               friction_query_
            );

            if (frictions_entities.size() == 0)
            {
               std::cout << "Friction system didn't find any entities that have a vector of friction constraints attached to them\n";
               return;
            }

            auto frictions = allocator.getComponent<constraintFrictions_t>(
               *frictions_entities.begin()
            );

            frictions->clear();

            const auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

            for (const auto & manifold : manifolds)
            {
               float mu_total = 2.f * (
                  (manifold.bodyAMu * manifold.bodyBMu) / (
                     manifold.bodyAMu + manifold.bodyBMu
                  )
               );

               if (mu_total < 1e-7f)
               {
                  continue;
               }

               oy::types::constraintFriction_t friction;
               friction.bodyIdA = manifold.bodyIdA;
               friction.bodyIdB = manifold.bodyIdB;
               friction.muTotal = mu_total;
               friction.unitNormal = manifold.unitNormal;

               if (friction.bodyIdA != oy::types::null_body_entity)
               {
                  friction.bodyAForce = forques[friction.bodyIdA]->appliedForce;
               }

               if (friction.bodyIdB != oy::types::null_body_entity)
               {
                  friction.bodyBForce = forques[friction.bodyIdB]->appliedForce;
               }

               for (unsigned int i = 0; i < manifold.numContacts; ++i)
               {
                  friction.bodyAContact += manifold.bodyAContacts[i] / manifold.numContacts;
                  friction.bodyBContact += manifold.bodyBContacts[i] / manifold.numContacts;
               }

               frictions->push_back(friction);
            }
         }

      private:
         trecs::query_t manifold_query_;

         trecs::query_t friction_query_;

   };
}

#endif
