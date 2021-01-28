#ifndef NARROWPHASE_SYSTEM_HEADER
#define NARROWPHASE_SYSTEM_HEADER

#include "allocator.hpp"

#include "slingshot_types.hpp"
#include "gjk.hpp"
#include "system.hpp"

#include <vector>

namespace oy
{
   template <typename ShapeA_T, typename ShapeB_T>
   class NarrowphaseSystem : public trecs::System
   {
      private:
         typedef oy::types::collisionCandidate_t<ShapeA_T, ShapeB_T> CollisionCandidate_T;
         typedef std::vector<CollisionCandidate_T> CollisionCandidates_T;
         typedef oy::types::collisionDetection_t<ShapeA_T, ShapeB_T> CollisionDetection_T;
         typedef std::vector<CollisionDetection_T> CollisionDetections_T;

      public:
         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<CollisionCandidates_T>();
            allocator.registerComponent<CollisionDetections_T>();
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
            CollisionDetections_T * detections = \
               allocator.getComponent<CollisionDetections_T>(pipeline_entity);

            if (candidates == nullptr)
            {
               std::cout << "Narrowphase candidates are null\n";
               return;
            }

            if (detections == nullptr)
            {
               std::cout << "Narrowphase detections are null\n";
               return;
            }

            detections->clear();

            for (const auto & candidate : *candidates)
            {
               gjk_result_t gjk_result = geometry::gjk::alg(
                  candidate.trans_A_to_W,
                  candidate.trans_B_to_W,
                  candidate.shape_a,
                  candidate.shape_b
               );

               if (!gjk_result.intersection)
               {
                  continue;
               }

               CollisionDetection_T detection;
               detection.initialize(candidate, gjk_result.minSimplex);
               detections->push_back(detection);
            }
         }

      private:
         trecs::query_t collision_pipeline_query_;
   };
}

#endif
