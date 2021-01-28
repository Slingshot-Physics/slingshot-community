#include "balljoint_widget.hpp"

#include "geometry.hpp"

#include "imgui.h"

trecs::uid_t BalljointConstraintWidget::addDefaultComponent(void)
{
   oy::types::constraintBalljoint_t default_balljoint;
   default_balljoint.parentLinkPoint[1] = 1.f;
   default_balljoint.childLinkPoint[1] = -1.f;

   return addBalljoint({-1, -1}, default_balljoint);
}

void BalljointConstraintWidget::deleteComponent(trecs::uid_t entity)
{
   auto balljoint_viz = allocator_.getComponent<balljoint_viz_ids_t>(entity);
   renderer_.deleteRenderable(balljoint_viz->linkPointMeshId);
   allocator_.removeEntity(entity);
}

void BalljointConstraintWidget::componentsUi(void)
{
   auto balljoints = allocator_.getComponents<oy::types::constraintBalljoint_t>();
   if (balljoints.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto balljoint_entities = allocator_.getQueryEntities(balljoint_query_);

   for (const auto balljoint_entity : balljoint_entities)
   {
      std::string balljoint_label("balljoint ");
      balljoint_label += std::to_string(balljoint_entity);

      oy::types::constraintBalljoint_t & balljoint = *balljoints[balljoint_entity];
      trecs::edge_t & edge = *edges[balljoint_entity];

      geometry::types::isometricTransform_t trans_A_to_W = getTransform(
         edge.nodeIdA
      );

      geometry::types::isometricTransform_t trans_B_to_W = getTransform(
         edge.nodeIdB
      );

      balljoint_viz_ids_t & viz_ids = *allocator_.getComponent<balljoint_viz_ids_t>(balljoint_entity);

      // Have to update the link position in the blind because the parent's
      // position might change when the balljoint UI isn't modified.
      Vector3 link_W = geometry::transform::forwardBound(trans_A_to_W, balljoint.parentLinkPoint);
      balljoint.childLinkPoint = geometry::transform::inverseBound(trans_B_to_W, link_W);

      renderer_.updateMeshTransform(
         viz_ids.linkPointMeshId,
         link_W,
         identityMatrix(),
         identityMatrix()
      );

      balljointUi(
         balljoint_entity,
         balljoint_label,
         edge,
         balljoint,
         viz_ids
      );
   }
}

trecs::uid_t BalljointConstraintWidget::addBalljoint(
   const oy::types::bodyLink_t body_link,
   const oy::types::constraintBalljoint_t & balljoint
)
{
   trecs::uid_t new_balljoint_uid = allocator_.addEntity(body_link.parentId, body_link.childId);
   allocator_.addComponent(new_balljoint_uid, balljoint);

   balljoint_viz_ids_t viz_id;
   viz_id.render = true;

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );
   viz_id.linkPointMeshId = renderer_.addMesh(temp_mesh_data, lavender, 0);

   allocator_.addComponent(new_balljoint_uid, viz_id);

   return new_balljoint_uid;
}

void BalljointConstraintWidget::balljointUi(
   const trecs::uid_t balljoint_uid,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::constraintBalljoint_t & balljoint,
   balljoint_viz_ids_t & viz_ids
)
{
   if (ImGui::TreeNode(label.c_str()))
   {
      const auto & rigid_body_entities = allocator_.getQueryEntities(rigid_body_query_);

      bool render_changed = ImGui::Checkbox("render", &viz_ids.render);

      trecs::uid_t excluded_body_id = (edge.nodeIdB > -1) ? edge.nodeIdB : -2;

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
         ImGui::Text("Link point A");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##1", &balljoint.parentLinkPoint[0], 0.1f);
         ImGui::PopItemWidth();

         Vector3 link_point_a = balljoint.parentLinkPoint;
         Vector3 link_point_b = balljoint.childLinkPoint;

         // These are for display only
         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Link point A body");
         ImGui::TableSetColumnIndex(1);
         // The push/pop item width calls get rid of the text labels on the
         // basic widgets, I think.
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##2", &link_point_a[0], 0.1f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Link point B body");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##3", &link_point_b[0], 0.1f);
         ImGui::PopItemWidth();

         ImGui::EndTable();
      }

      if (render_changed)
      {
         if (viz_ids.render)
         {
            renderer_.enableMesh(viz_ids.linkPointMeshId);
         }
         else
         {
            renderer_.disableMesh(viz_ids.linkPointMeshId);
         }
      }

      if (ImGui::Button("Delete balljoint"))
      {
         deleteComponent(balljoint_uid);
      }

      ImGui::TreePop();
   }
}
