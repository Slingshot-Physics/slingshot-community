#include "rotation_1d_widget.hpp"

#include "attitudeutils.hpp"
#include "geometry.hpp"
#include "geometry_type_converters.hpp"

#include "imgui.h"

trecs::uid_t Rotation1dConstraintWidget::addDefaultComponent(void)
{
   oy::types::constraintRotation1d_t default_rotation_1d;
   default_rotation_1d.parentAxis[2] = 1.f;
   default_rotation_1d.childAxis[1] = 1.f;

   return addRotation1d({-1, -1}, default_rotation_1d);
}

void Rotation1dConstraintWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<rotation_1d_viz_ids_t>(entity);
   renderer_.deleteRenderable(viz_ids->childAxisMeshId);
   renderer_.deleteRenderable(viz_ids->parentAxisMeshId);
   allocator_.removeEntity(entity);
}

void Rotation1dConstraintWidget::componentsUi(void)
{
   auto rotation_1ds = allocator_.getComponents<oy::types::constraintRotation1d_t>();
   if (rotation_1ds.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto rotation_1d_entities = allocator_.getQueryEntities(rotation_1d_query_);

   bool render_all_pressed = false;
   bool hide_all_pressed = false;

   ImGui::PushID("rotation 1d");
   if (rotation_1d_entities.size() > 0)
   {
      render_all_pressed = ImGui::Button("Render all");
      hide_all_pressed = ImGui::Button("Hide all");
   }
   ImGui::PopID();

   for (const auto rotation_1d_entity : rotation_1d_entities)
   {
      std::string rotation_1d_label("1d rotation constraint ");
      rotation_1d_label += std::to_string(rotation_1d_entity);

      oy::types::constraintRotation1d_t & rotation_1d = *rotation_1ds[rotation_1d_entity];
      trecs::edge_t & edge = *edges[rotation_1d_entity];

      auto & viz_ids = *allocator_.getComponent<rotation_1d_viz_ids_t>(rotation_1d_entity);

      if (render_all_pressed)
      {
         viz_ids.render = true;
         renderer_.enableMesh(viz_ids.childAxisMeshId);
         renderer_.enableMesh(viz_ids.parentAxisMeshId);
      }
      if (hide_all_pressed)
      {
         viz_ids.render = false;
         renderer_.disableMesh(viz_ids.childAxisMeshId);
         renderer_.disableMesh(viz_ids.parentAxisMeshId);
      }

      geometry::types::isometricTransform_t trans_A_to_W = getTransform(
         edge.nodeIdA
      );

      geometry::types::isometricTransform_t trans_B_to_W = getTransform(
         edge.nodeIdB
      );

      // Force body B's link point to be the closest point on the constraint
      // plane to body B's center of mass position.

      const Vector3 parent_axis_W = geometry::transform::forwardUnbound(
         trans_A_to_W, rotation_1d.parentAxis
      );

      const Vector3 child_axis_W = geometry::transform::forwardUnbound(
         trans_B_to_W, rotation_1d.childAxis
      );

      Matrix33 R_parent_axis_up = makeVectorUp(parent_axis_W);

      Matrix33 R_child_axis_up = makeVectorUp(child_axis_W);

      renderer_.updateMeshTransform(
         viz_ids.parentAxisMeshId,
         trans_A_to_W.translate,
         R_parent_axis_up.transpose(),
         identityMatrix()
      );

      renderer_.updateMeshTransform(
         viz_ids.childAxisMeshId,
         trans_B_to_W.translate,
         R_child_axis_up.transpose(),
         identityMatrix()
      );

      rotation1dUi(
         rotation_1d_entity,
         rotation_1d_label,
         edge,
         rotation_1d,
         viz_ids
      );
   }
}

trecs::uid_t Rotation1dConstraintWidget::addRotation1d(
   const oy::types::bodyLink_t body_link,
   const oy::types::constraintRotation1d_t & rotation_1d
)
{
   trecs::uid_t new_rotation_1d_uid = allocator_.addEntity(body_link.parentId, body_link.childId);
   allocator_.addComponent(new_rotation_1d_uid, rotation_1d);

   rotation_1d_viz_ids_t viz_ids;
   viz_ids.render = true;

   geometry::types::triangleMesh_t cylinder = geometry::mesh::loadDefaultShapeMesh(
      geometry::types::enumShape_t::CYLINDER
   );
   geometry::mesh::scaleMesh(Vector3(0.05f, 0.05f, 5.f), cylinder);

   data_triangleMesh_t temp_mesh_data;

   geometry::converters::to_pod(cylinder, &temp_mesh_data);
   viz_ids.parentAxisMeshId = renderer_.addMesh(
      temp_mesh_data, red, 0
   );
   viz_ids.childAxisMeshId = renderer_.addMesh(
      temp_mesh_data, green, 0
   );

   allocator_.addComponent(new_rotation_1d_uid, viz_ids);

   return new_rotation_1d_uid;
}

void Rotation1dConstraintWidget::rotation1dUi(
   const trecs::uid_t rotation_1d_uid,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::constraintRotation1d_t & rotation_1d,
   rotation_1d_viz_ids_t & viz_ids
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
         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Parent axis unit vector");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##1", &rotation_1d.parentAxis[0], 0.01f);
         ImGui::PopItemWidth();

         rotation_1d.parentAxis.Normalize();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Child axis unit vector");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##2", &rotation_1d.childAxis[0], 0.01f);
         ImGui::PopItemWidth();

         rotation_1d.childAxis.Normalize();

         ImGui::EndTable();
      }

      if (render_changed)
      {
         if (viz_ids.render)
         {
            renderer_.enableMesh(viz_ids.parentAxisMeshId);
            renderer_.enableMesh(viz_ids.childAxisMeshId);
         }
         else
         {
            renderer_.disableMesh(viz_ids.parentAxisMeshId);
            renderer_.disableMesh(viz_ids.childAxisMeshId);
         }
      }

      if (ImGui::Button("Delete rotation 1D constraint"))
      {
         deleteComponent(rotation_1d_uid);
      }

      ImGui::TreePop();
   }
}
