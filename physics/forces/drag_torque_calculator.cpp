#include "drag_torque_calculator.hpp"

#include "drag_torque.hpp"

namespace oy
{
   void DragTorqueCalculator::registerComponents(trecs::Allocator & allocator) const
   {
      allocator.registerComponent<oy::types::rigidBody_t>();
      allocator.registerComponent<oy::types::generalizedForce_t>();
      allocator.registerComponent<oy::types::torqueDrag_t>();
   }

   void DragTorqueCalculator::registerQueries(trecs::Allocator & allocator)
   {
      drag_query_ = allocator.addArchetypeQuery<oy::types::torqueDrag_t>();
   }

   void DragTorqueCalculator::update(trecs::Allocator & allocator) const
   {
      const auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
      const auto drags = allocator.getComponents<oy::types::torqueDrag_t>();
      const auto edges = allocator.getComponents<trecs::edge_t>();
      auto forques = allocator.getComponents<oy::types::generalizedForce_t>();

      const auto & edge_entities = allocator.getQueryEntities(drag_query_);

      for (const auto entity : edge_entities)
      {
         const trecs::edge_t & edge = *edges[entity];
         if (edge.edgeId == -1 || edge.nodeIdB == oy::types::null_body_entity)
         {
            if (edge.edgeId == -1)
            {
               std::cout << "Bad edge id: " << entity << "\n";
            }
            if (edge.nodeIdB == oy::types::null_body_entity)
            {
               std::cout << "Drag torque body ID should in edge.nodeIdA should be NEQ -1\n";
            }
            continue;
         }

         const oy::types::rigidBody_t & child_body = *bodies[edge.nodeIdB];

         const oy::types::torqueDrag_t * drag = drags[entity];

         oy::types::generalizedForce_t & forque = *forques[edge.nodeIdB];

         if (drag == nullptr)
         {
            std::cout << "null drag torque at entity id: " << entity << "\n";
            continue;
         }

         oy::torques::drag::evaluateTorque(*drag, child_body, forque);
      }
   }
}
