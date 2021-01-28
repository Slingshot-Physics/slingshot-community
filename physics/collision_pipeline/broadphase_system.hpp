#ifndef BROADPHASE_SYSTEM_TEMPLATE_HEADER
#define BROADPHASE_SYSTEM_TEMPLATE_HEADER

#include "allocator.hpp"

#include "slingshot_types.hpp"
#include "sat.hpp"
#include "system.hpp"

#include <type_traits>
#include <vector>

namespace oy
{
   template <typename ShapeA_T, typename ShapeB_T>
   class BroadphaseSystem : public trecs::System
   {
      private:

         typedef oy::types::collisionCandidate_t<ShapeA_T, ShapeB_T> CollisionCandidate_T;
         typedef oy::types::collisionDetection_t<ShapeA_T, ShapeB_T> CollisionDetection_T;
         typedef oy::types::collisionGeometry_t<ShapeA_T, ShapeB_T> CollisionGeometry_T;

      public:

         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<oy::types::isometricCollider_t>();
            allocator.registerComponent<oy::types::rigidBody_t>();
            allocator.registerComponent<geometry::types::aabb_t>();
            allocator.registerComponent<geometry::types::shape_t>();
            allocator.registerComponent<oy::types::DynamicBody>();
            allocator.registerComponent<oy::types::StationaryBody>();
            allocator.registerComponent<ShapeA_T>();
            allocator.registerComponent<ShapeB_T>();

            // Output component types
            allocator.registerComponent<std::vector<CollisionCandidate_T> >();
            allocator.registerComponent<std::vector<CollisionDetection_T> >();
            allocator.registerComponent<std::vector<CollisionGeometry_T> >();
            allocator.registerComponent<std::vector<oy::types::collisionContactManifold_t> >();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            shape_a_rigid_body_query_ = allocator.addArchetypeQuery<
               ShapeA_T,
               geometry::types::aabb_t,
               oy::types::rigidBody_t,
               oy::types::isometricCollider_t
            >();

            shape_b_rigid_body_query_ = allocator.addArchetypeQuery<
               ShapeB_T,
               geometry::types::aabb_t,
               oy::types::rigidBody_t,
               oy::types::isometricCollider_t
            >();

            collision_pipeline_query_ = allocator.addArchetypeQuery<
               std::vector<CollisionCandidate_T>
            >();
         }

         void initialize(trecs::Allocator & allocator) override
         {
            // The entity that holds the collision pipeline for this particular
            // shape pair.
            trecs::uid_t pipeline_entity = allocator.addEntity(-1);

            std::vector<CollisionCandidate_T> temp_candidates;
            if (!allocator.addComponent(pipeline_entity, temp_candidates))
            {
               std::cout << "Error adding vector of candidates from broad phase\n";
            }

            std::vector<CollisionDetection_T> temp_detections;
            if (!allocator.addComponent(pipeline_entity, temp_detections))
            {
               std::cout << "Error adding vector of detections from broad phase\n";
            }

            std::vector<CollisionGeometry_T> temp_geometries;
            if (!allocator.addComponent(pipeline_entity, temp_geometries))
            {
               std::cout << "Error adding vector of geometries from broad phase\n";
            }
         }

         void update(trecs::Allocator & allocator) const
         {
            if (std::is_same<ShapeA_T, ShapeB_T>::value)
            {
               sameShapesUpdate(allocator);
            }
            else
            {
               differentShapesUpdate(allocator);
            }
         }

      private:

         trecs::query_t shape_a_rigid_body_query_;

         trecs::query_t shape_b_rigid_body_query_;

         trecs::query_t rigid_body_query_;

         trecs::query_t collision_pipeline_query_;

         void differentShapesUpdate(trecs::Allocator & allocator) const
         {
            const auto shape_a_entities = allocator.getQueryEntities(shape_a_rigid_body_query_);
            const auto shape_b_entities = allocator.getQueryEntities(shape_b_rigid_body_query_);

            if (shape_a_entities.size() == 0 || shape_b_entities.size() == 0)
            {
               return;
            }

            auto colliders = allocator.getComponents<oy::types::isometricCollider_t>();
            auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
            auto type_a_shapes = allocator.getComponents<ShapeA_T>();
            auto type_b_shapes = allocator.getComponents<ShapeB_T>();
            auto aabbs = allocator.getComponents<geometry::types::aabb_t>();

            trecs::uid_t pipeline_entity = *allocator.getQueryEntities(collision_pipeline_query_).begin();

            auto candidates = allocator.getComponent<std::vector<CollisionCandidate_T> >(pipeline_entity);

            if (candidates != nullptr)
            {
               candidates->clear();
            }

            for (const auto & shape_a_entity : shape_a_entities)
            {
               const oy::types::rigidBody_t * body_a = bodies[shape_a_entity];
               const oy::types::isometricCollider_t * collider_a = colliders[shape_a_entity];
               const geometry::types::aabb_t * aabb_a = aabbs[shape_a_entity];
               const ShapeA_T * shape_a = type_a_shapes[shape_a_entity];
               bool body_a_stationary = allocator.hasComponent<oy::types::StationaryBody>(shape_a_entity);

               if (!collider_a->enabled)
               {
                  continue;
               }

               for (const auto & shape_b_entity : shape_b_entities)
               {
                  if (shape_a_entity == shape_b_entity)
                  {
                     continue;
                  }

                  const oy::types::rigidBody_t * body_b = bodies[shape_b_entity];
                  const oy::types::isometricCollider_t * collider_b = colliders[shape_b_entity];
                  const geometry::types::aabb_t * aabb_b = aabbs[shape_b_entity];
                  const ShapeB_T * shape_b = type_b_shapes[shape_b_entity];
                  bool body_b_stationary = allocator.hasComponent<oy::types::StationaryBody>(shape_b_entity);

                  if (
                     (body_a_stationary && body_b_stationary) ||
                     (!collider_b->enabled)
                  )
                  {
                     continue;
                  }

                  if (geometry::collisions::aabbAabb(*aabb_a, *aabb_b))
                  {
                     CollisionCandidate_T candidate;

                     candidate.nodeIdA = shape_a_entity;
                     candidate.nodeIdB = shape_b_entity;
                     candidate.trans_A_to_W.rotate = body_a->ql2b.rotationMatrix().transpose();
                     candidate.trans_A_to_W.translate = body_a->linPos;
                     candidate.trans_B_to_W.rotate = body_b->ql2b.rotationMatrix().transpose();
                     candidate.trans_B_to_W.translate = body_b->linPos;
                     candidate.shape_a = *shape_a;
                     candidate.shape_b = *shape_b;

                     candidates->push_back(candidate);
                  }
               }
            }
         }

         void sameShapesUpdate(trecs::Allocator & allocator) const
         {
            const auto shape_entities = allocator.getQueryEntities(shape_a_rigid_body_query_);

            if (shape_entities.size() == 0 || shape_entities.size() == 0)
            {
               return;
            }

            auto colliders = allocator.getComponents<oy::types::isometricCollider_t>();
            auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
            auto type_a_shapes = allocator.getComponents<ShapeA_T>();
            auto type_b_shapes = allocator.getComponents<ShapeB_T>();
            auto aabbs = allocator.getComponents<geometry::types::aabb_t>();

            trecs::uid_t pipeline_entity = *allocator.getQueryEntities(collision_pipeline_query_).begin();

            auto candidates = allocator.getComponent<std::vector<CollisionCandidate_T> >(pipeline_entity);

            if (candidates != nullptr)
            {
               candidates->clear();
            }

            for (auto iter_a = shape_entities.begin(); iter_a != shape_entities.end(); ++iter_a)
            {
               const trecs::uid_t shape_a_entity = *iter_a;
               const oy::types::isometricCollider_t * collider_a = colliders[shape_a_entity];

               if (!collider_a->enabled)
               {
                  continue;
               }

               const oy::types::rigidBody_t * body_a = bodies[shape_a_entity];
               const geometry::types::aabb_t * aabb_a = aabbs[shape_a_entity];
               const ShapeA_T * shape_a = type_a_shapes[shape_a_entity];
               bool body_a_stationary = allocator.hasComponent<oy::types::StationaryBody>(shape_a_entity);

               for (auto iter_b = iter_a; iter_b != shape_entities.end(); ++iter_b)
               {
                  const trecs::uid_t shape_b_entity = *iter_b;
                  if (shape_a_entity == shape_b_entity)
                  {
                     continue;
                  }

                  const oy::types::rigidBody_t * body_b = bodies[shape_b_entity];
                  const oy::types::isometricCollider_t * collider_b = colliders[shape_b_entity];
                  const geometry::types::aabb_t * aabb_b = aabbs[shape_b_entity];
                  const ShapeB_T * shape_b = type_b_shapes[shape_b_entity];
                  bool body_b_stationary = allocator.hasComponent<oy::types::StationaryBody>(shape_b_entity);

                  if (
                     (body_a_stationary && body_b_stationary) ||
                     (!collider_b->enabled)
                  )
                  {
                     continue;
                  }

                  if (geometry::collisions::aabbAabb(*aabb_a, *aabb_b))
                  {
                     CollisionCandidate_T candidate;

                     candidate.nodeIdA = shape_a_entity;
                     candidate.nodeIdB = shape_b_entity;
                     candidate.trans_A_to_W.rotate = body_a->ql2b.rotationMatrix().transpose();
                     candidate.trans_A_to_W.translate = body_a->linPos;
                     candidate.trans_B_to_W.rotate = body_b->ql2b.rotationMatrix().transpose();
                     candidate.trans_B_to_W.translate = body_b->linPos;
                     candidate.shape_a = *shape_a;
                     candidate.shape_b = *shape_b;

                     candidates->push_back(candidate);
                  }
               }
            }
         }

   };
}

#endif
