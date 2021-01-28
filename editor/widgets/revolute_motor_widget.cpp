#include "revolute_motor_widget.hpp"

#include "attitudeutils.hpp"
#include "geometry.hpp"
#include "geometry_type_converters.hpp"

#include "imgui.h"

trecs::uid_t RevoluteMotorConstraintWidget::addDefaultComponent(void)
{
   oy::types::constraintRevoluteMotor_t default_revolute_motor;
   default_revolute_motor.parentAxis[2] = 1.f;
   default_revolute_motor.childAxis[2] = 1.f;
   default_revolute_motor.angularSpeed = 0.f;
   default_revolute_motor.maxTorque = 1.f;

   return addRevoluteMotor({-1, -1}, default_revolute_motor);
}

void RevoluteMotorConstraintWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<revolute_motor_viz_ids_t>(entity);
   renderer_.deleteRenderable(viz_ids->axisParentMeshId);
   renderer_.deleteRenderable(viz_ids->axisChildMeshId);
   allocator_.removeEntity(entity);
}

void RevoluteMotorConstraintWidget::componentsUi(void)
{
   auto revolute_motors = allocator_.getComponents<oy::types::constraintRevoluteMotor_t>();
   if (revolute_motors.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto revolute_motor_entities = allocator_.getQueryEntities(revolute_motor_query_);

   for (const auto revolute_motor_entity : revolute_motor_entities)
   {
      std::string revolute_motor_label("revolute motor ");
      revolute_motor_label += std::to_string(revolute_motor_entity);

      oy::types::constraintRevoluteMotor_t & revolute_motor = *revolute_motors[revolute_motor_entity];
      trecs::edge_t & edge = *edges[revolute_motor_entity];

      revolute_motor_viz_ids_t & viz_ids = *allocator_.getComponent<revolute_motor_viz_ids_t>(revolute_motor_entity);

      geometry::types::isometricTransform_t trans_A_to_W = getTransform(
         edge.nodeIdA
      );

      Vector3 axis_a_W = geometry::transform::forwardUnbound(
         trans_A_to_W, revolute_motor.parentAxis
      );
      Matrix33 R_W_to_axis_A = makeVectorUp(axis_a_W);

      renderer_.updateMeshTransform(
         viz_ids.axisParentMeshId,
         trans_A_to_W.translate,
         R_W_to_axis_A.transpose(),
         identityMatrix()
      );

      geometry::types::isometricTransform_t trans_B_to_W = getTransform(
         edge.nodeIdB
      );

      Vector3 axis_b_W = geometry::transform::forwardUnbound(
         trans_B_to_W, revolute_motor.childAxis
      );
      Matrix33 R_W_to_axis_B = makeVectorUp(axis_b_W);

      renderer_.updateMeshTransform(
         viz_ids.axisChildMeshId,
         trans_B_to_W.translate,
         R_W_to_axis_B.transpose(),
         identityMatrix()
      );

      revoluteMotorUi(
         revolute_motor_entity,
         revolute_motor_label,
         edge,
         revolute_motor,
         viz_ids
      );
   }
}

trecs::uid_t RevoluteMotorConstraintWidget::addRevoluteMotor(
   const oy::types::bodyLink_t body_link,
   const oy::types::constraintRevoluteMotor_t & revolute_motor
)
{
   trecs::uid_t new_revolute_motor_uid = allocator_.addEntity(body_link.parentId, body_link.childId);
   allocator_.addComponent(new_revolute_motor_uid, revolute_motor);

   revolute_motor_viz_ids_t viz_ids;
   viz_ids.render = true;

   geometry::types::triangleMesh_t cylinder = geometry::mesh::loadDefaultShapeMesh(
      geometry::types::enumShape_t::CYLINDER
   );
   geometry::mesh::scaleMesh(Vector3(0.05f, 0.05f, 5.f), cylinder);
   data_triangleMesh_t cylinder_data;
   geometry::converters::to_pod(cylinder, &cylinder_data);

   viz_ids.axisParentMeshId = renderer_.addMesh(cylinder_data, red, 0);
   viz_ids.axisChildMeshId = renderer_.addMesh(cylinder_data, green, 0);

   allocator_.addComponent(new_revolute_motor_uid, viz_ids);

   return new_revolute_motor_uid;
}

void RevoluteMotorConstraintWidget::revoluteMotorUi(
   const trecs::uid_t revolute_motor_uid,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::constraintRevoluteMotor_t & revolute_motor,
   revolute_motor_viz_ids_t & viz_ids
)
{
   if (ImGui::TreeNode(label.c_str()))
   {
      const auto & rigid_body_entities = allocator_.getQueryEntities(rigid_body_query_);

      bool render_changed = ImGui::Checkbox("render", &viz_ids.render);

      int excluded_body_id = (edge.nodeIdB > -1) ? edge.nodeIdB : -2;

      bodyIdComboBox(
         parent_body_box_label_,
         excluded_body_id,
         rigid_body_entities,
         edge.nodeIdA
      );

      excluded_body_id = (edge.nodeIdA > -1) ? edge.nodeIdA : -2;

      bodyIdComboBox(
         child_body_box_label_,
         excluded_body_id,
         rigid_body_entities,
         edge.nodeIdB
      );

      if (ImGui::BeginTable("fields", 2, ImGuiTableFlags_Borders))
      {
         ImGui::TableSetupColumn("Field name", ImGuiTableColumnFlags_WidthFixed);
         ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthStretch);

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Drive speed");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##1angularspeed", &revolute_motor.angularSpeed, 0.1f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Max torque");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##2maxtorque", &revolute_motor.maxTorque, 0.1f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Axis body A");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##4", &revolute_motor.parentAxis[0], 0.01f);
         revolute_motor.parentAxis.Normalize();
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Axis body B");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##5", &revolute_motor.childAxis[0], 0.01f);
         revolute_motor.childAxis.Normalize();
         ImGui::PopItemWidth();

         ImGui::EndTable();
      }

      if (render_changed)
      {
         if (viz_ids.render)
         {
            renderer_.enableMesh(viz_ids.axisParentMeshId);
            renderer_.enableMesh(viz_ids.axisChildMeshId);
         }
         else
         {
            renderer_.disableMesh(viz_ids.axisParentMeshId);
            renderer_.disableMesh(viz_ids.axisChildMeshId);
         }
      }

      if (ImGui::Button("Delete revolute motor"))
      {
         deleteComponent(revolute_motor_uid);
      }

      ImGui::TreePop();
   }
}
