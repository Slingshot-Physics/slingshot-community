#ifndef SAT_SYSTEM_HEADER
#define SAT_SYSTEM_HEADER

#include "allocator.hpp"

#include "slingshot_types.hpp"
#include "system.hpp"
#include "sat.hpp"

#include <vector>

namespace oy
{
   template <typename ShapeA_T, typename ShapeB_T>
   class SatSystem : public trecs::System
   {
      private:
         typedef oy::types::collisionCandidate_t<ShapeA_T, ShapeB_T> CollisionCandidate_T;
         typedef std::vector<CollisionCandidate_T> CollisionCandidates_T;
         typedef oy::types::collisionGeometry_t<ShapeA_T, ShapeB_T> CollisionGeometry_T;
         typedef std::vector<CollisionGeometry_T> CollisionGeometries_T;

      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<CollisionGeometries_T>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            collision_pipeline_query_ = allocator.addArchetypeQuery<
               CollisionCandidates_T
            >();
         }

         void update(trecs::Allocator & allocator) const
         {
            trecs::uid_t pipeline_entity = *allocator.getQueryEntities(collision_pipeline_query_).begin();

            CollisionCandidates_T * candidates = \
               allocator.getComponent<CollisionCandidates_T>(pipeline_entity);
            CollisionGeometries_T * geometries = \
               allocator.getComponent<CollisionGeometries_T>(pipeline_entity);

            if (candidates == nullptr)
            {
               std::cout << "SAT candidates are null\n";
               return;
            }

            if (geometries == nullptr)
            {
               std::cout << "SAT geometries are null\n";
               return;
            }

            geometries->clear();

            for (const auto & candidate : *candidates)
            {
               geometry::types::satResult_t sat_result = geometry::collisions::sat(
                  candidate.trans_A_to_W,
                  candidate.trans_B_to_W,
                  candidate.shape_a,
                  candidate.shape_b
               );

               if (!sat_result.collision)
               {
                  continue;
               }

               CollisionGeometry_T geometry;
               geometry.initialize(candidate, sat_result);

               geometries->push_back(geometry);
            }
         }

      private:
         trecs::query_t collision_pipeline_query_;

   };
}

#endif
