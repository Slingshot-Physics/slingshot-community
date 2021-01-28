#ifndef DRAG_FORCE_WIDGET_HEADER
#define DRAG_FORCE_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class DragForceWidget : public IComponentWidget
{

   struct drag_force_viz_ids_t
   {
      // I have no idea what would even be rendered
   };

   public:
      DragForceWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<drag_force_viz_ids_t>();
         allocator_.registerComponent<oy::types::forceDrag_t>();

         drag_query_ = allocator.addArchetypeQuery<oy::types::forceDrag_t>();
      }

      ~DragForceWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addDragForce(
         const oy::types::bodyLink_t body_link,
         const oy::types::forceDrag_t & component
      );

   private:
      trecs::query_t drag_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void dragForceUi(
         const trecs::uid_t entity,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::forceDrag_t & component,
         drag_force_viz_ids_t & viz_ids
      );
};

#endif
