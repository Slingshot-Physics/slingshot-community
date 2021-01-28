#ifndef COLLISION_CONSTRAINT_SYSTEM_HEADER
#define COLLISION_CONSTRAINT_SYSTEM_HEADER

#include "allocator.hpp"

#include "slingshot_types.hpp"
#include "system.hpp"

#include <vector>

namespace oy
{
   class CollisionSystem : public trecs::System
   {
      private:
         typedef std::vector<oy::types::constraintCollision_t> constraintCollisions_t;
         typedef std::vector<oy::types::collisionContactManifold_t> collisionManifolds_t;

      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<constraintCollisions_t>();
            allocator.registerComponent<collisionManifolds_t>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            manifold_query_ = allocator.addArchetypeQuery<
               collisionManifolds_t
            >();

            collision_query_ = allocator.addArchetypeQuery<
               constraintCollisions_t
            >();
         }

         void initialize(trecs::Allocator & allocator) override
         {
            // The entity that will hold the vector of collision constraints.
            trecs::uid_t constraint_collisions_entity = allocator.addEntity();

            constraintCollisions_t temp_collisions;
            if (!allocator.addComponent(constraint_collisions_entity, temp_collisions))
            {
               std::cout << "Error adding vector of collision constraints from the collision constraint system\n";
            }
         }

         void update(trecs::Allocator & allocator) const
         {
            const auto manifold_entities = allocator.getQueryEntities(
               manifold_query_
            );

            if (manifold_entities.size() == 0)
            {
               std::cout << "Collision system didn't find any entities that have a vector of contact manifolds attached to them\n";
               return;
            }

            const auto & manifolds = *allocator.getComponent<collisionManifolds_t>(
               *manifold_entities.begin()
            );

            const auto collisions_entities = allocator.getQueryEntities(
               collision_query_
            );

            if (collisions_entities.size() == 0)
            {
               std::cout << "Collision system didn't find any entities that have a vector of collision constraints attached to them\n";
               return;
            }

            auto collisions = allocator.getComponent<constraintCollisions_t>(
               *collisions_entities.begin()
            );

            collisions->clear();

            for (const auto & manifold : manifolds)
            {
               for (unsigned int i = 0; i < manifold.numContacts; ++i)
               {
                  const Vector3 & contact_a_W = manifold.bodyAContacts[i];
                  const Vector3 & contact_b_W = manifold.bodyBContacts[i];
                  const Vector3 & contact_normal_W = manifold.unitNormal;

                  // Don't include contacts that don't violate the constraint.
                  if ((contact_b_W - contact_a_W).dot(contact_normal_W) >= 0.f)
                  {
                     continue;
                  }

                  oy::types::constraintCollision_t collision;
                  collision.bodyIdA = manifold.bodyIdA;
                  collision.bodyIdB = manifold.bodyIdB;
                  collision.bodyAContact = contact_a_W;
                  collision.bodyBContact = contact_b_W;
                  collision.unitNormal = manifold.unitNormal;
                  collision.restitution = 0.5f * (
                     manifold.bodyARestitution + manifold.bodyBRestitution
                  );

                  collisions->push_back(collision);
               }
            }
         }

      private:
         trecs::query_t manifold_query_;

         trecs::query_t collision_query_;

   };
}

#endif
