#include "drag_force_widget.hpp"

#include "geometry.hpp"

#include "imgui.h"

trecs::uid_t DragForceWidget::addDefaultComponent(void)
{
   oy::types::forceDrag_t default_drag;
   default_drag.linearDragCoeff = -0.04f;
   default_drag.quadraticDragCoeff = 0.f;

   return addDragForce({-1, -1}, default_drag);
}

void DragForceWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<drag_force_viz_ids_t>(entity);
   (void)viz_ids;
   allocator_.removeComponent<oy::types::forceDrag_t>(entity);
}

void DragForceWidget::componentsUi(void)
{
   auto forces = allocator_.getComponents<oy::types::forceDrag_t>();
   if (forces.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto force_entities = allocator_.getQueryEntities(drag_query_);

   for (const auto force_entity : force_entities)
   {
      std::string force_label("drag force ");
      force_label += std::to_string(force_entity);

      oy::types::forceDrag_t & force = *forces[force_entity];
      trecs::edge_t & edge = *edges[force_entity];

      auto & viz_ids = *allocator_.getComponent<drag_force_viz_ids_t>(force_entity);

      dragForceUi(
         force_entity,
         force_label,
         edge,
         force,
         viz_ids
      );
   }
}

trecs::uid_t DragForceWidget::addDragForce(
   const oy::types::bodyLink_t body_link,
   const oy::types::forceDrag_t & component
)
{
   trecs::uid_t new_force_entity = allocator_.addEntity(
      body_link.parentId, body_link.childId
   );
   allocator_.addComponent(new_force_entity, component);

   drag_force_viz_ids_t viz_ids;

   allocator_.addComponent(new_force_entity, viz_ids);

   return new_force_entity;
}

void DragForceWidget::dragForceUi(
   const trecs::uid_t entity,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::forceDrag_t & component,
   drag_force_viz_ids_t & viz_ids
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
