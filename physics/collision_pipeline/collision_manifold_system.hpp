#ifndef COLLISION_MANIFOLD_SYSTEM_HEADER
#define COLLISION_MANIFOLD_SYSTEM_HEADER

#include "allocator.hpp"

#include "geometry_types.hpp"
#include "shape_features.hpp"
#include "system.hpp"

#include "collision_contacts.hpp"

#include <vector>

namespace oy
{
   template <typename ShapeA_T, typename ShapeB_T>
   class CollisionManifoldSystem : public trecs::System
   {
      typedef oy::types::collisionGeometry_t<ShapeA_T, ShapeB_T> CollisionGeometry_T;
      typedef std::vector<CollisionGeometry_T> CollisionGeometries_T;
      typedef std::vector<oy::types::collisionContactManifold_t> collisionManifolds_t;

      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            // allocator.registerComponent<oy::types::categoryCollisionManifold_t>();
            allocator.registerComponent<CollisionGeometries_T>();
            allocator.registerComponent<collisionManifolds_t>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            collision_pipeline_query_ = allocator.addArchetypeQuery<
               CollisionGeometries_T
            >();

            contact_manifold_query_ = allocator.addArchetypeQuery<
               collisionManifolds_t
            >();
         }

         void initialize(trecs::Allocator & allocator) override
         {
            if (allocator.getQueryEntities(contact_manifold_query_).size() != 0)
            {
               return;
            }

            // The entity that holds the manifold output for all shape pairs.
            trecs::uid_t manifold_entity = allocator.addEntity();

            collisionManifolds_t temp_manifolds;
            if (!allocator.addComponent(manifold_entity, temp_manifolds))
            {
               std::cout << "Error adding vector of collision manifolds from collision manifold system\n";
            }
         }

         void update(trecs::Allocator & allocator) const
         {
            const auto & pipeline_entities = allocator.getQueryEntities(
               collision_pipeline_query_
            );

            if (pipeline_entities.size() == 0)
            {
               std::cout << "Collision manifold system found zero entities with the collision pipeline category on them\n";
               return;
            }

            trecs::uid_t pipeline_entity = *pipeline_entities.begin();

            const auto & manifold_entities = allocator.getQueryEntities(
               contact_manifold_query_
            );

            if (manifold_entities.size() == 0)
            {
               std::cout << "Collision manifold system found zero entities with the contact manifold category on them\n";
               return;
            }

            trecs::uid_t manifold_entity = *manifold_entities.begin();

            CollisionGeometries_T * geometries = \
               allocator.getComponent<CollisionGeometries_T>(pipeline_entity);

            if (geometries == nullptr)
            {
               std::cout << "Manifold geometries are null\n";
               return;
            }

            collisionManifolds_t * manifolds = \
               allocator.getComponent<collisionManifolds_t>(manifold_entity);

            if (manifolds == nullptr)
            {
               std::cout << "Manifold manifolds are null\n";
               return;
            }

            auto colliders = allocator.getComponents<oy::types::isometricCollider_t>();

            for (const auto & geometry : *geometries)
            {
               oy::types::contactGeometry_t contact_geometry;
               contact_geometry.contactNormal = geometry.contactNormal;
               contact_geometry.bodyAContactPoint = geometry.bodyAContactPoint;
               contact_geometry.bodyBContactPoint = geometry.bodyBContactPoint;

               oy::types::collisionContactManifold_t contact_manifold = oy::collision::calculateContactManifold(
                  geometry.shape_a,
                  geometry.shape_b,
                  geometry.trans_A_to_W,
                  geometry.trans_B_to_W,
                  contact_geometry
               );

               const oy::types::isometricCollider_t * collider_a = colliders[geometry.nodeIdA];
               const oy::types::isometricCollider_t * collider_b = colliders[geometry.nodeIdB];

               contact_manifold.bodyIdA = geometry.nodeIdA;
               contact_manifold.bodyAMu = collider_a->mu;
               contact_manifold.bodyARestitution = collider_a->restitution;

               contact_manifold.bodyIdB = geometry.nodeIdB;
               contact_manifold.bodyBMu = collider_b->mu;
               contact_manifold.bodyBRestitution = collider_b->restitution;

               contact_manifold.unitNormal = geometry.contactNormal;

               manifolds->push_back(contact_manifold);
            }
         }

      private:
         trecs::query_t collision_pipeline_query_;

         trecs::query_t contact_manifold_query_;

   };
}

#endif
