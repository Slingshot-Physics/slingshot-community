#ifndef TORSIONAL_FRICTION_CONSTRAINT_SYSTEM_HEADER
#define TORSIONAL_FRICTION_CONSTRAINT_SYSTEM_HEADER

#include "allocator.hpp"

#include "slingshot_types.hpp"
#include "system.hpp"

#include <vector>

namespace oy
{
   class TorsionalFrictionSystem : public trecs::System
   {
      private:
         typedef std::vector<oy::types::constraintTorsionalFriction_t> constraintTorsionalFrictions_t;
         typedef std::vector<oy::types::collisionContactManifold_t> collisionManifolds_t;

      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<constraintTorsionalFrictions_t>();
            allocator.registerComponent<collisionManifolds_t>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            manifold_query_ = allocator.addArchetypeQuery<
               collisionManifolds_t
            >();

            friction_query_ = allocator.addArchetypeQuery<
               constraintTorsionalFrictions_t
            >();
         }

         void initialize(trecs::Allocator & allocator) override
         {
            // The entity that will hold the vector of friction constraints.
            trecs::uid_t constraint_frictions_entity = allocator.addEntity();

            constraintTorsionalFrictions_t temp_frictions;
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

            auto frictions = allocator.getComponent<constraintTorsionalFrictions_t>(
               *frictions_entities.begin()
            );

            frictions->clear();

            const auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

            const auto bodies = allocator.getComponents<oy::types::rigidBody_t>();

            for (const auto & manifold : manifolds)
            {
               float mu_total = 2.f * (
                  (manifold.bodyAMu * manifold.bodyBMu) / (
                     manifold.bodyAMu + manifold.bodyBMu
                  )
               );

               if (
                  (manifold.numContacts == 1) ||
                  (mu_total < 1e-7f)
               )
               {
                  continue;
               }

               float ang_vel_rel_normal = 0.f;

               const auto body_a = bodies[manifold.bodyIdA];
               const auto body_b = bodies[manifold.bodyIdB];

               if (
                  (manifold.bodyIdA != oy::types::null_body_entity) &&
                  (manifold.bodyIdB != oy::types::null_body_entity)
               )
               {
                  ang_vel_rel_normal = (
                     body_a->ql2b.conjugateSandwich(body_a->angVel) -\
                     body_b->ql2b.conjugateSandwich(body_b->angVel)
                  ).dot(manifold.unitNormal);
               }
               else if (
                  (manifold.bodyIdA != oy::types::null_body_entity) &&
                  (manifold.bodyIdB == oy::types::null_body_entity)
               )
               {
                  ang_vel_rel_normal = (
                     body_a->ql2b.conjugateSandwich(body_a->angVel)
                  ).dot(manifold.unitNormal);
               }
               else if (
                  (manifold.bodyIdA == oy::types::null_body_entity) &&
                  (manifold.bodyIdB != oy::types::null_body_entity)
               )
               {
                  ang_vel_rel_normal = (
                     body_b->ql2b.conjugateSandwich(body_b->angVel)
                  ).dot(manifold.unitNormal);
               }

               if (fabs(ang_vel_rel_normal) < 1e-5f)
               {
                  continue;
               }

               oy::types::constraintTorsionalFriction_t friction;
               friction.bodyIdA = manifold.bodyIdA;
               friction.bodyIdB = manifold.bodyIdB;
               friction.muTotal = mu_total;
               friction.unitNormal = manifold.unitNormal;
               friction.leverArmLength = 0.f;

               if (friction.bodyIdA != oy::types::null_body_entity)
               {
                  friction.bodyAForce = forques[friction.bodyIdA]->appliedForce;
               }

               if (friction.bodyIdB != oy::types::null_body_entity)
               {
                  friction.bodyBForce = forques[friction.bodyIdB]->appliedForce;
               }

               // Calculate the center of the contact points in world
               // coordinates
               Vector3 contact_center_W;

               for (unsigned int i = 0; i < manifold.numContacts; ++i)
               {
                  contact_center_W += manifold.bodyAContacts[i] / manifold.numContacts;
               }

               // Use the center of the contact points to find the length of
               // the longest moment arm.
               for (unsigned int i = 0; i < manifold.numContacts; ++i)
               {
                  const float temp_arm_length = (
                     manifold.bodyAContacts[i] - contact_center_W
                  ).magnitude();

                  if (temp_arm_length > friction.leverArmLength)
                  {
                     friction.leverArmLength = temp_arm_length;
                  }
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
