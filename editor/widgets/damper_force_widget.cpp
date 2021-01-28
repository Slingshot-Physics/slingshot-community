#include "damper_force_widget.hpp"

#include "geometry.hpp"

#include "imgui.h"

trecs::uid_t DamperForceWidget::addDefaultComponent(void)
{
   oy::types::forceVelocityDamper_t default_damper;
   default_damper.damperCoeff = -1.f;
   default_damper.parentLinkPoint[2] = 1.f;
   default_damper.childLinkPoint[2] = 1.f;

   return addDamperForce({-1, -1}, default_damper);
}

void DamperForceWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<velocity_damper_viz_ids_t>(entity);
   renderer_.deleteRenderable(viz_ids->linkPointParentMeshId);
   renderer_.deleteRenderable(viz_ids->linkPointChildMeshId);
   allocator_.removeComponent<oy::types::forceVelocityDamper_t>(entity);
}

void DamperForceWidget::componentsUi(void)
{
   auto dampers = allocator_.getComponents<oy::types::forceVelocityDamper_t>();
   if (dampers.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto damper_entities = allocator_.getQueryEntities(damper_query_);

   for (const auto damper_entity : damper_entities)
   {
      std::string damper_label("velocity damper ");
      damper_label += std::to_string(damper_entity);

      oy::types::forceVelocityDamper_t & damper = *dampers[damper_entity];
      trecs::edge_t & edge = *edges[damper_entity];

      auto & viz_ids = *allocator_.getComponent<velocity_damper_viz_ids_t>(damper_entity);

      geometry::types::isometricTransform_t trans_A_to_W = getTransform(
         edge.nodeIdA
      );

      Vector3 link_a_W = geometry::transform::forwardBound(
         trans_A_to_W, damper.parentLinkPoint
      );

      renderer_.updateMeshTransform(
         viz_ids.linkPointParentMeshId,
         link_a_W,
         identityMatrix(),
         identityMatrix()
      );

      geometry::types::isometricTransform_t trans_B_to_W = getTransform(
         edge.nodeIdB
      );

      Vector3 link_b_W = geometry::transform::forwardBound(
         trans_B_to_W, damper.childLinkPoint
      );

      renderer_.updateMeshTransform(
         viz_ids.linkPointChildMeshId,
         link_b_W,
         identityMatrix(),
         identityMatrix()
      );

      damperForceUi(
         damper_entity,
         damper_label,
         edge,
         damper,
         viz_ids
      );
   }
}

trecs::uid_t DamperForceWidget::addDamperForce(
   const oy::types::bodyLink_t body_link,
   const oy::types::forceVelocityDamper_t & damper
)
{
   trecs::uid_t new_damper_entity = allocator_.addEntity(
      body_link.parentId, body_link.childId
   );
   allocator_.addComponent<oy::types::forceVelocityDamper_t>(
      new_damper_entity, damper
   );

   velocity_damper_viz_ids_t viz_ids;
   viz_ids.render = true;

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );

   viz_ids.linkPointParentMeshId = renderer_.addMesh(
      temp_mesh_data, cyan, 0
   );

   viz_ids.linkPointChildMeshId = renderer_.addMesh(
      temp_mesh_data, cyan, 0
   );

   allocator_.addComponent(new_damper_entity, viz_ids);

   return new_damper_entity;
}

void DamperForceWidget::damperForceUi(
   const trecs::uid_t velocity_damper_uid,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::forceVelocityDamper_t & velocity_damper,
   velocity_damper_viz_ids_t & viz_ids
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
         ImGui::Text("%s", "link point A");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##linkpointA", &(velocity_damper.parentLinkPoint[0]), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("%s", "link point B");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##linkpointB", &(velocity_damper.childLinkPoint[0]), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("%s", "velocity_damper coefficient");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##velocity_dampercoeff", &(velocity_damper.damperCoeff), 0.01f, -1000.f, -1e-5f);
         ImGui::PopItemWidth();

         ImGui::EndTable();
      }

      if (render_changed)
      {
         if (viz_ids.render)
         {
            renderer_.enableMesh(viz_ids.linkPointParentMeshId);
            renderer_.enableMesh(viz_ids.linkPointChildMeshId);
         }
         else
         {
            renderer_.disableMesh(viz_ids.linkPointParentMeshId);
            renderer_.disableMesh(viz_ids.linkPointChildMeshId);
         }
      }

      if (ImGui::Button("Delete velocity damper"))
      {
         deleteComponent(velocity_damper_uid);
      }

      ImGui::TreePop();
   }
}
