#ifndef CONSTANT_FORCE_WIDGET_HEADER
#define CONSTANT_FORCE_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class ConstantForceWidget : public IComponentWidget
{

   struct constant_force_viz_ids_t
   {
      bool render;
      int linkPointBodyMeshId;
      int forceLineMeshId;
   };

   public:
      ConstantForceWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<constant_force_viz_ids_t>();
         allocator_.registerComponent<oy::types::forceConstant_t>();

         constant_force_query_ = allocator.addArchetypeQuery<oy::types::forceConstant_t>();
      }

      ~ConstantForceWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addConstantForce(
         const oy::types::bodyLink_t body_link,
         const oy::types::forceConstant_t & component
      );

   private:
      trecs::query_t constant_force_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void constantForceUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::forceConstant_t & component,
         constant_force_viz_ids_t & viz_ids
      );
};

#endif
