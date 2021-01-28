#include "spring_force_widget.hpp"

#include "geometry.hpp"

#include "imgui.h"

trecs::uid_t SpringForceWidget::addDefaultComponent(void)
{
   oy::types::forceSpring_t default_spring;
   default_spring.parentLinkPoint[2] = 1.f;
   default_spring.childLinkPoint[2] = 1.f;
   default_spring.restLength = 1.f;
   default_spring.springCoeff = -1.f;

   return addSpringForce({-1, -1}, default_spring);
}

void SpringForceWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<spring_viz_ids_t>(entity);
   renderer_.deleteRenderable(viz_ids->linkPointParentMeshId);
   renderer_.deleteRenderable(viz_ids->linkPointChildMeshId);
   renderer_.deleteRenderable(viz_ids->springLineMeshId);
   allocator_.removeComponent<oy::types::forceSpring_t>(entity);
}

void SpringForceWidget::componentsUi(void)
{
   auto springs = allocator_.getComponents<oy::types::forceSpring_t>();
   if (springs.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto spring_entities = allocator_.getQueryEntities(spring_query_);

   for (const auto spring_entity : spring_entities)
   {
      std::string spring_label("spring ");
      spring_label += std::to_string(spring_entity);

      oy::types::forceSpring_t & spring = *springs[spring_entity];
      trecs::edge_t & edge = *edges[spring_entity];

      spring_viz_ids_t & viz_ids = *allocator_.getComponent<spring_viz_ids_t>(spring_entity);

      geometry::types::isometricTransform_t trans_A_to_W = getTransform(
         edge.nodeIdA
      );

      Vector3 link_a_W = geometry::transform::forwardBound(
         trans_A_to_W, spring.parentLinkPoint
      );

      renderer_.updateMeshTransform(
         viz_ids.linkPointParentMeshId,
         trans_A_to_W.translate,
         identityMatrix(),
         identityMatrix()
      );

      geometry::types::isometricTransform_t trans_B_to_W = getTransform(
         edge.nodeIdB
      );

      Vector3 link_b_W = geometry::transform::forwardBound(
         trans_B_to_W, spring.childLinkPoint
      );

      renderer_.updateMeshTransform(
         viz_ids.linkPointChildMeshId,
         trans_B_to_W.translate,
         identityMatrix(),
         identityMatrix()
      );

      Vector3 links_W[2] = {link_a_W, link_b_W};
      renderer_.updateSegment(viz_ids.springLineMeshId, 2, links_W);

      springForceUi(
         spring_entity,
         spring_label,
         edge,
         spring,
         viz_ids
      );
   }
}

trecs::uid_t SpringForceWidget::addSpringForce(
   const oy::types::bodyLink_t body_link,
   const oy::types::forceSpring_t & spring
)
{
   trecs::uid_t new_spring_uid = allocator_.addEntity(
      body_link.parentId, body_link.childId
   );
   allocator_.addComponent(new_spring_uid, spring);

   spring_viz_ids_t viz_ids;

   viz_ids.render = true;

   data_triangleMesh_t temp_mesh_data = geometry::mesh::loadDefaultShapeMeshData(
      geometry::types::enumShape_t::SPHERE, 0.1f
   );

   viz_ids.linkPointParentMeshId = renderer_.addMesh(
      temp_mesh_data, 0
   );

   viz_ids.linkPointChildMeshId = renderer_.addMesh(
      temp_mesh_data, 0
   );

   Vector3 spring_line[2] = {
      {0.f, 0.f, 1.f},
      {0.f, 0.f, 1.f},
   };

   viz_ids.springLineMeshId = renderer_.addSegment(
      2, spring_line, red
   );

   allocator_.addComponent(new_spring_uid, viz_ids);

   return new_spring_uid;
}

void SpringForceWidget::springForceUi(
   const trecs::uid_t spring_uid,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::forceSpring_t & spring,
   spring_viz_ids_t & viz_ids
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
         ImGui::DragFloat3("##linkpointA", &(spring.parentLinkPoint[0]), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("%s", "link point B");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##linkpointB", &(spring.childLinkPoint[0]), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("%s", "rest length");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##restlength", &(spring.restLength), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("%s", "spring coefficient");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##springcoeff", &(spring.springCoeff), 0.01f, -1000.f, -1e-5f);
         ImGui::PopItemWidth();

         ImGui::EndTable();
      }

      if (render_changed)
      {
         if (viz_ids.render)
         {
            renderer_.enableMesh(viz_ids.linkPointParentMeshId);
            renderer_.enableMesh(viz_ids.linkPointChildMeshId);
            renderer_.enableMesh(viz_ids.springLineMeshId);
         }
         else
         {
            renderer_.disableMesh(viz_ids.linkPointParentMeshId);
            renderer_.disableMesh(viz_ids.linkPointChildMeshId);
            renderer_.disableMesh(viz_ids.springLineMeshId);
         }
      }

      if (ImGui::Button("Delete spring"))
      {
         deleteComponent(spring_uid);
      }

      ImGui::TreePop();
   }
}
