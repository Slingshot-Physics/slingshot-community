#include "constant_force_widget.hpp"

#include "geometry.hpp"

#include "imgui.h"

trecs::uid_t ConstantForceWidget::addDefaultComponent(void)
{
   oy::types::forceConstant_t default_force;
   default_force.acceleration[2] = -9.8f;
   default_force.forceFrame = oy::types::enumFrame_t::GLOBAL;

   return addConstantForce({-1, -1}, default_force);
}

void ConstantForceWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<constant_force_viz_ids_t>(entity);
   renderer_.deleteRenderable(viz_ids->linkPointBodyMeshId);
   renderer_.deleteRenderable(viz_ids->forceLineMeshId);
   allocator_.removeComponent<oy::types::forceConstant_t>(entity);
}

void ConstantForceWidget::componentsUi(void)
{
   auto forces = allocator_.getComponents<oy::types::forceConstant_t>();
   if (forces.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto force_entities = allocator_.getQueryEntities(constant_force_query_);

   bool render_all_pressed = false;
   bool hide_all_pressed = false;

   ImGui::PushID("constant force");
   if (force_entities.size() > 0)
   {
      render_all_pressed = ImGui::Button("Render all");
      hide_all_pressed = ImGui::Button("Hide all");
   }
   ImGui::PopID();

   for (const auto force_entity : force_entities)
   {
      std::string force_label("constant force ");
      force_label += std::to_string(force_entity);

      oy::types::forceConstant_t & force = *forces[force_entity];
      trecs::edge_t & edge = *edges[force_entity];

      auto & viz_ids = *allocator_.getComponent<constant_force_viz_ids_t>(force_entity);

      if (render_all_pressed)
      {
         viz_ids.render = true;
         renderer_.enableMesh(viz_ids.linkPointBodyMeshId);
         renderer_.enableMesh(viz_ids.forceLineMeshId);
      }
      if (hide_all_pressed)
      {
         viz_ids.render = false;
         renderer_.disableMesh(viz_ids.linkPointBodyMeshId);
         renderer_.disableMesh(viz_ids.forceLineMeshId);
      }

      geometry::types::isometricTransform_t trans_B_to_W = getTransform(
         edge.nodeIdB
      );

      Vector3 link_b_W = geometry::transform::forwardBound(
         trans_B_to_W, force.childLinkPoint
      );

      renderer_.updateMeshTransform(
         viz_ids.linkPointBodyMeshId,
         link_b_W,
         identityMatrix(),
         identityMatrix()
      );

      Vector3 links_W[2] = {link_b_W, link_b_W + force.acceleration};

      if (force.forceFrame == oy::types::enumFrame_t::BODY)
      {
         links_W[1] = geometry::transform::forwardBound(trans_B_to_W, force.acceleration);
      }

      renderer_.updateSegment(viz_ids.forceLineMeshId, 2, links_W);

      constantForceUi(
         force_entity,
         force_label,
         edge,
         force,
         viz_ids
      );
   }
}

trecs::uid_t ConstantForceWidget::addConstantForce(
   const oy::types::bodyLink_t body_link,
   const oy::types::forceConstant_t & constant_force
)
{
   trecs::uid_t new_force_entity = allocator_.addEntity(
      body_link.parentId, body_link.childId
   );
   allocator_.addComponent(new_force_entity, constant_force);

   constant_force_viz_ids_t viz_ids;

   viz_ids.render = true;

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );

   viz_ids.linkPointBodyMeshId = renderer_.addMesh(
      temp_mesh_data, 0
   );

   Vector3 force_line[2] = {
      {0.f, 0.f, 1.f},
      {0.f, 0.f, 1.f},
   };

   viz_ids.forceLineMeshId = renderer_.addSegment(
      2, force_line, red
   );

   allocator_.addComponent(new_force_entity, viz_ids);

   return new_force_entity;
}

void ConstantForceWidget::constantForceUi(
   const trecs::uid_t force_entity,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::forceConstant_t & force,
   constant_force_viz_ids_t & viz_ids
)
{
   if (ImGui::TreeNode(label.c_str()))
   {
      const auto & rigid_body_entities = allocator_.getQueryEntities(rigid_body_query_);

      bool render_changed = ImGui::Checkbox("render", &viz_ids.render);

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
         ImGui::Text("%s", "Link point");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##linkpoint", &(force.childLinkPoint[0]), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("%s", "Acceleration");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##force_val", &(force.acceleration[0]), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("%s", "Frame");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         int force_frame = static_cast<int>(force.forceFrame);
         ImGui::RadioButton("Global", &force_frame, static_cast<int>(oy::types::enumFrame_t::GLOBAL));
         ImGui::SameLine();
         ImGui::RadioButton("Body", &force_frame, static_cast<int>(oy::types::enumFrame_t::BODY));
         force.forceFrame = static_cast<oy::types::enumFrame_t>(force_frame);
         ImGui::PopItemWidth();

         ImGui::EndTable();
      }

      if (render_changed)
      {
         if (viz_ids.render)
         {
            renderer_.enableMesh(viz_ids.linkPointBodyMeshId);
            renderer_.enableMesh(viz_ids.forceLineMeshId);
         }
         else
         {
            renderer_.disableMesh(viz_ids.linkPointBodyMeshId);
            renderer_.disableMesh(viz_ids.forceLineMeshId);
         }
      }

      if (ImGui::Button("Delete constant force"))
      {
         deleteComponent(force_entity);
      }

      ImGui::TreePop();
   }
}