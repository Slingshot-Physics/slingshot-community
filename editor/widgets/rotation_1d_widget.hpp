#ifndef ROTATION_1D_CONSTRAINT_WIDGET_HEADER
#define ROTATION_1D_CONSTRAINT_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class Rotation1dConstraintWidget : public IComponentWidget
{

   struct rotation_1d_viz_ids_t
   {
      bool render;
      int parentAxisMeshId;
      int childAxisMeshId;
   };

   public:
      Rotation1dConstraintWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<rotation_1d_viz_ids_t>();
         allocator_.registerComponent<oy::types::constraintRotation1d_t>();

         rotation_1d_query_ = allocator.addArchetypeQuery<oy::types::constraintRotation1d_t>();
      }

      ~Rotation1dConstraintWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addRotation1d(
         const oy::types::bodyLink_t body_link,
         const oy::types::constraintRotation1d_t & component
      );

   private:
      trecs::query_t rotation_1d_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void rotation1dUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::constraintRotation1d_t & component,
         rotation_1d_viz_ids_t & viz_ids
      );
};

#endif
