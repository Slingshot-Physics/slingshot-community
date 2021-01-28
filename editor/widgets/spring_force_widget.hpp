#ifndef SPRING_FORCE_WIDGET_HEADER
#define SPRING_FORCE_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class SpringForceWidget : public IComponentWidget
{

   struct spring_viz_ids_t
   {
      bool render;
      int linkPointParentMeshId;
      int linkPointChildMeshId;
      int springLineMeshId;
   };

   public:
      SpringForceWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<spring_viz_ids_t>();
         allocator_.registerComponent<oy::types::forceSpring_t>();

         spring_query_ = allocator.addArchetypeQuery<oy::types::forceSpring_t>();
      }

      ~SpringForceWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addSpringForce(
         const oy::types::bodyLink_t body_link,
         const oy::types::forceSpring_t & component
      );

   private:
      trecs::query_t spring_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void springForceUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::forceSpring_t & component,
         spring_viz_ids_t & viz_ids
      );
};

#endif
