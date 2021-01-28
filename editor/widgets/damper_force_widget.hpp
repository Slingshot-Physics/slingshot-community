#ifndef DAMPER_FORCE_WIDGET_HEADER
#define DAMPER_FORCE_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class DamperForceWidget : public IComponentWidget
{

   struct velocity_damper_viz_ids_t
   {
      bool render;
      int linkPointParentMeshId;
      int linkPointChildMeshId;
   };

   public:
      DamperForceWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<velocity_damper_viz_ids_t>();
         allocator_.registerComponent<oy::types::forceVelocityDamper_t>();

         damper_query_ = allocator.addArchetypeQuery<oy::types::forceVelocityDamper_t>();
      }

      ~DamperForceWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addDamperForce(
         const oy::types::bodyLink_t body_link,
         const oy::types::forceVelocityDamper_t & component
      );

   private:
      trecs::query_t damper_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void damperForceUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::forceVelocityDamper_t & component,
         velocity_damper_viz_ids_t & viz_ids
      );
};

#endif
