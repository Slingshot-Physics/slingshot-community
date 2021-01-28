#ifndef GEAR_CONSTRAINT_WIDGET_HEADER
#define GEAR_CONSTRAINT_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class GearConstraintWidget : public IComponentWidget
{

   struct gear_viz_ids_t
   {
      bool render;
      int axisParentMeshId;
      int axisChildMeshId;
   };

   public:
      GearConstraintWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<gear_viz_ids_t>();
         allocator_.registerComponent<oy::types::constraintGear_t>();

         rigid_body_query_ = allocator.addArchetypeQuery<oy::types::rigidBody_t>();
         gear_query_ = allocator.addArchetypeQuery<oy::types::constraintGear_t>();
      }

      ~GearConstraintWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addGear(
         const oy::types::bodyLink_t body_link,
         const oy::types::constraintGear_t & component
      );

   private:
      trecs::query_t gear_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void gearUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::constraintGear_t & component,
         gear_viz_ids_t & viz_ids
      );
};

#endif
