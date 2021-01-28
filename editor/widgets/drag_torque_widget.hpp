#ifndef DRAG_TORQUE_WIDGET_HEADER
#define DRAG_TORQUE_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class DragTorqueWidget : public IComponentWidget
{

   struct drag_torque_viz_ids_t
   {
      // I have no idea what would even be rendered
   };

   public:
      DragTorqueWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<drag_torque_viz_ids_t>();
         allocator_.registerComponent<oy::types::torqueDrag_t>();

         drag_query_ = allocator.addArchetypeQuery<oy::types::torqueDrag_t>();
      }

      ~DragTorqueWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addDragTorque(
         const oy::types::bodyLink_t body_link,
         const oy::types::torqueDrag_t & component
      );

   private:
      trecs::query_t drag_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void dragTorqueUi(
         const trecs::uid_t entity,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::torqueDrag_t & component,
         drag_torque_viz_ids_t & viz_ids
      );
};

#endif
