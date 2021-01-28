#ifndef COLLISION_GEOMETRY_SYSTEM_HEADER
#define COLLISION_GEOMETRY_SYSTEM_HEADER

#include "allocator.hpp"

#include "epa.hpp"
#include "system.hpp"

#include <cmath>
#include <vector>

namespace oy
{
   template <typename ShapeA_T, typename ShapeB_T, typename Termination_T>
   class CollisionGeometrySystem : public trecs::System
   {
      private:
         typedef oy::types::collisionDetection_t<ShapeA_T, ShapeB_T> CollisionDetection_T;
         typedef std::vector<CollisionDetection_T> CollisionDetections_T;
         typedef oy::types::collisionGeometry_t<ShapeA_T, ShapeB_T> CollisionGeometry_T;
         typedef std::vector<CollisionGeometry_T> CollisionGeometries_T;

      public:
         void setTerminationCriteria(
            const Termination_T & epa_termination_criteria
         )
         {
            epa_termination_criteria_ = epa_termination_criteria;
         }

         void registerComponents(trecs::Allocator & allocator) const override
         {
            allocator.registerComponent<CollisionDetections_T>();
            allocator.registerComponent<CollisionGeometries_T>();
         }

         void registerQueries(trecs::Allocator & allocator) override
         {
            collision_pipeline_query_ = allocator.addArchetypeQuery<
               CollisionDetections_T
            >();
         }

         void update(trecs::Allocator & allocator) const
         {
            trecs::uid_t pipeline_entity = *allocator.getQueryEntities(collision_pipeline_query_).begin();

            CollisionDetections_T * detections = \
               allocator.getComponent<CollisionDetections_T>(pipeline_entity);
            CollisionGeometries_T * geometries = \
               allocator.getComponent<CollisionGeometries_T>(pipeline_entity);

            if (detections == nullptr)
            {
               std::cout << "Geometry detections are null\n";
               return;
            }

            if (geometries == nullptr)
            {
               std::cout << "Geometry geometries are null\n";
               return;
            }

            geometries->clear();

            for (const auto & detection : *detections)
            {
               geometry::types::epaResult_t epa_result = calculateGeometry(
                  detection.trans_A_to_W,
                  detection.trans_B_to_W,
                  detection.shape_a,
                  detection.shape_b,
                  detection.simplex
               );

               if (epa_result.collided && epa_result.p.magnitude() > 1e-4f)
               {
                  epa_result.p = alignScaleCollisionNormal(
                     detection.trans_A_to_W.translate,
                     detection.trans_B_to_W.translate,
                     epa_result.p
                  );

                  CollisionGeometry_T geometry;
                  geometry.initialize(detection, epa_result);

                  geometries->push_back(geometry);
               }
            }
         }

      private:
         trecs::query_t collision_pipeline_query_;

         Termination_T epa_termination_criteria_;

         template <typename Transform_T>
         geometry::types::epaResult_t calculateGeometry(
            const Transform_T & trans_A_to_W,
            const Transform_T & trans_B_to_W,
            const ShapeA_T & shape_a,
            const ShapeB_T & shape_b,
            const geometry::types::minkowskiDiffSimplex_t gjk_simplex
         ) const
         {
            geometry::types::minkowskiDiffSimplex_t tetra_simplex = \
               gjk_simplex;

            if (tetra_simplex.numVerts < 4)
            {
               geometry::epa::expandGjkSimplex(
                  trans_A_to_W, trans_B_to_W, shape_a, shape_b, tetra_simplex
               );
            }

            bool degenerate_tetrahedron = geometry::plane::pointCoplanar(
               tetra_simplex.verts[0],
               tetra_simplex.verts[1],
               tetra_simplex.verts[2],
               tetra_simplex.verts[3],
               1e-3f
            );

            geometry::types::epaResult_t epa_result;
            if (degenerate_tetrahedron)
            {
               epa_result.collided = false;
               return epa_result;
            }

            epa_result = geometry::epa::alg(
               trans_A_to_W,
               trans_B_to_W,
               tetra_simplex,
               shape_a,
               shape_b,
               epa_termination_criteria_
            );

            return epa_result;
         }

         // Returns a collision normal that is a unit vector pointing from body
         // A to body B.
         Vector3 alignScaleCollisionNormal(
            const Vector3 & pos_a_W,
            const Vector3 & pos_b_W,
            Vector3 collision_normal_W
         ) const
         {
            const Vector3 d_ab_W = pos_b_W - pos_a_W;
            const float sign = (
               std::signbit(d_ab_W.dot(collision_normal_W)) ? -1.f : 1.f
            );
            collision_normal_W *= sign;
            collision_normal_W.Normalize();

            return collision_normal_W;
         }

   };
}

#endif
