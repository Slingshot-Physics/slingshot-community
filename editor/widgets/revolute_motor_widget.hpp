#ifndef REVOLUTE_MOTOR_CONSTRAINT_WIDGET_HEADER
#define REVOLUTE_MOTOR_CONSTRAINT_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <string>

class RevoluteMotorConstraintWidget : public IComponentWidget
{

   struct revolute_motor_viz_ids_t
   {
      bool render;
      int axisParentMeshId;
      int axisChildMeshId;
   };

   public:
      RevoluteMotorConstraintWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
         , parent_body_box_label_("Parent body")
         , child_body_box_label_("Child body")
      {
         allocator_.registerComponent<revolute_motor_viz_ids_t>();
         allocator_.registerComponent<oy::types::constraintRevoluteMotor_t>();

         revolute_motor_query_ = allocator.addArchetypeQuery<oy::types::constraintRevoluteMotor_t>();
      }

      ~RevoluteMotorConstraintWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addRevoluteMotor(
         const oy::types::bodyLink_t body_link,
         const oy::types::constraintRevoluteMotor_t & component
      );

   private:
      trecs::query_t revolute_motor_query_;

      const std::string parent_body_box_label_;

      const std::string child_body_box_label_;

      void revoluteMotorUi(
         const trecs::uid_t component_uid,
         const std::string & label,
         trecs::edge_t & edge,
         oy::types::constraintRevoluteMotor_t & component,
         revolute_motor_viz_ids_t & viz_ids
      );
};

#endif
