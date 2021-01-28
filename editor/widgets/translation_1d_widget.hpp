#ifndef TRANSLATION_1D_CONSTRAINT_WIDGET_HEADER
#define TRANSLATION_1D_CONSTRAINT_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class Translation1dConstraintWidget : public IComponentWidget
{

   struct planar_joint_viz_ids_t
   {
      bool render;
      int parentLinkPointMeshId;
      int childLinkPointMeshId;
      int parentAxisMeshId;
   };

   public:
      Translation1dConstraintWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<planar_joint_viz_ids_t>();
         allocator_.registerComponent<oy::types::constraintTranslation1d_t>();

         planar_joint_query_ = allocator.addArchetypeQuery<oy::types::constraintTranslation1d_t>();
      }

      ~Translation1dConstraintWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addTranslation1d(
         const oy::types::bodyLink_t body_link,
         const oy::types::constraintTranslation1d_t & component
      );

   private:
      trecs::query_t planar_joint_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void translation1dUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::constraintTranslation1d_t & component,
         planar_joint_viz_ids_t & viz_ids
      );
};

#endif
