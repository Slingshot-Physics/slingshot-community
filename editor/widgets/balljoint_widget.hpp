#ifndef BALL_JOINT_CONSTRAINT_WIDGET_HEADER
#define BALL_JOINT_CONSTRAINT_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class BalljointConstraintWidget : public IComponentWidget
{

   struct balljoint_viz_ids_t
   {
      bool render;
      int linkPointMeshId;
   };

   public:
      BalljointConstraintWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<balljoint_viz_ids_t>();
         allocator_.registerComponent<oy::types::constraintBalljoint_t>();

         balljoint_query_ = allocator.addArchetypeQuery<oy::types::constraintBalljoint_t>();
      }

      ~BalljointConstraintWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addBalljoint(
         const oy::types::bodyLink_t body_link,
         const oy::types::constraintBalljoint_t & component
      );

   private:
      trecs::query_t balljoint_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void balljointUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::constraintBalljoint_t & component,
         balljoint_viz_ids_t & viz_ids
      );
};

#endif
