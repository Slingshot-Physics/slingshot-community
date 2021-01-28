#ifndef REVOLUTE_JOINT_CONSTRAINT_WIDGET_HEADER
#define REVOLUTE_JOINT_CONSTRAINT_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class RevoluteJointConstraintWidget : public IComponentWidget
{

   struct revolute_joint_viz_ids_t
   {
      bool render;
      int linkPointParentMeshIds[2];
      int linkPointChildMeshIds[2];
   };

   public:
      RevoluteJointConstraintWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<revolute_joint_viz_ids_t>();
         allocator_.registerComponent<oy::types::constraintRevoluteJoint_t>();

         revolute_joint_query_ = allocator.addArchetypeQuery<oy::types::constraintRevoluteJoint_t>();
      }

      ~RevoluteJointConstraintWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addRevoluteJoint(
         const oy::types::bodyLink_t body_link,
         const oy::types::constraintRevoluteJoint_t & component
      );

   private:
      trecs::query_t revolute_joint_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void revoluteJointUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::constraintRevoluteJoint_t & component,
         revolute_joint_viz_ids_t & viz_ids
      );
};

#endif
