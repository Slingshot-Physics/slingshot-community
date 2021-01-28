#ifndef AABB_CALCULATOR_TEMPLATE_HEADER
#define AABB_CALCULATOR_TEMPLATE_HEADER

#include "allocator.hpp"

#include "aabb_hull.hpp"
#include "transform.hpp"

#include "system.hpp"

namespace oy
{
   template <typename Shape_T>
   class AabbCalculator : public trecs::System
   {
      public:

         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<oy::types::isometricCollider_t>();
            allocator.registerComponent<oy::types::rigidBody_t>();
            allocator.registerComponent<geometry::types::aabb_t>();
            allocator.registerComponent<Shape_T>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            rigid_body_query_ = allocator.addArchetypeQuery<
               Shape_T,
               geometry::types::aabb_t,
               oy::types::rigidBody_t,
               oy::types::isometricCollider_t
            >();
         }

         void update(trecs::Allocator & allocator) const
         {
            auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
            auto aabbs = allocator.getComponents<geometry::types::aabb_t>();
            auto shapes = allocator.getComponents<Shape_T>();

            const auto & node_entities = allocator.getQueryEntities(rigid_body_query_);

            for (const auto entity : node_entities)
            {
               const oy::types::rigidBody_t * temp_body = bodies[entity];

               geometry::types::isometricTransform_t trans_C_to_W;
               trans_C_to_W.rotate = temp_body->ql2b.rotationMatrix().transpose();
               trans_C_to_W.translate = temp_body->linPos;

               const Shape_T * temp_shape = shapes[entity];

               geometry::types::aabb_t * aabb = aabbs[entity];

               *aabb = geometry::aabbHull(trans_C_to_W, *temp_shape);
            }
         }

      private:
         trecs::query_t rigid_body_query_;
   };

}

#endif
