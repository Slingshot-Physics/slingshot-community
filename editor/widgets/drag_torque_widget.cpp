#include "drag_torque_widget.hpp"

#include "geometry.hpp"

#include "imgui.h"

trecs::uid_t DragTorqueWidget::addDefaultComponent(void)
{
   oy::types::torqueDrag_t default_drag;
   default_drag.linearDragCoeff = -0.04f;
   default_drag.quadraticDragCoeff = 0.f;

   return addDragTorque({-1, -1}, default_drag);
}

void DragTorqueWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<drag_torque_viz_ids_t>(entity);
   (void)viz_ids;
   allocator_.removeComponent<oy::types::torqueDrag_t>(entity);
}

void DragTorqueWidget::componentsUi(void)
{
   auto torques = allocator_.getComponents<oy::types::torqueDrag_t>();
   if (torques.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto torque_entities = allocator_.getQueryEntities(drag_query_);

   for (const auto torque_entity : torque_entities)
   {
      std::string torque_label("drag torque ");
      torque_label += std::to_string(torque_entity);

      oy::types::torqueDrag_t & torque = *torques[torque_entity];
      trecs::edge_t & edge = *edges[torque_entity];

      auto & viz_ids = *allocator_.getComponent<drag_torque_viz_ids_t>(torque_entity);

      dragTorqueUi(
         torque_entity,
         torque_label,
         edge,
         torque,
         viz_ids
      );
   }
}

trecs::uid_t DragTorqueWidget::addDragTorque(
   const oy::types::bodyLink_t body_link,
   const oy::types::torqueDrag_t & component
)
{
   trecs::uid_t new_torque_entity = allocator_.addEntity(
      body_link.parentId, body_link.childId
   );
   allocator_.addComponent(new_torque_entity, component);

   drag_torque_viz_ids_t viz_ids;

   allocator_.addComponent(new_torque_entity, viz_ids);

   return new_torque_entity;
}

void DragTorqueWidget::dragTorqueUi(
   const trecs::uid_t entity,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::torqueDrag_t & component,
   drag_torque_viz_ids_t & viz_ids
)
{
   (void)viz_ids;
   if (ImGui::TreeNode(label.c_str()))
   {
      const auto & rigid_body_entities = allocator_.getQueryEntities(rigid_body_query_);

      int excluded_body_id = (edge.nodeIdA > -1) ? edge.nodeIdA : -2;

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
         ImGui::Text("%s", "Linear drag");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##linear_drag", &(component.linearDragCoeff), 0.0001f, -30.f, 0.f, "%0.6f");
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("%s", "Quadratic drag");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##quadratic_drag", &(component.quadraticDragCoeff), 0.0001f, -30.f, 0.f, "%0.6f");
         ImGui::PopItemWidth();

         ImGui::EndTable();
      }

      if (ImGui::Button("Delete drag torque"))
      {
         deleteComponent(entity);
      }

      ImGui::TreePop();
   }
}
