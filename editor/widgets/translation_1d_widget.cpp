#include "translation_1d_widget.hpp"

#include "attitudeutils.hpp"
#include "geometry.hpp"
#include "geometry_type_converters.hpp"

#include "imgui.h"

trecs::uid_t Translation1dConstraintWidget::addDefaultComponent(void)
{
   oy::types::constraintTranslation1d_t default_planar_joint;
   default_planar_joint.parentAxis[2] = 1.f;
   default_planar_joint.parentLinkPoint[2] = -1.f;
   default_planar_joint.childLinkPoint[2] = 1.f;

   return addTranslation1d({-1, -1}, default_planar_joint);
}

void Translation1dConstraintWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<planar_joint_viz_ids_t>(entity);
   renderer_.deleteRenderable(viz_ids->parentLinkPointMeshId);
   renderer_.deleteRenderable(viz_ids->childLinkPointMeshId);
   renderer_.deleteRenderable(viz_ids->parentAxisMeshId);
   allocator_.removeEntity(entity);
}

void Translation1dConstraintWidget::componentsUi(void)
{
   auto planar_joints = allocator_.getComponents<oy::types::constraintTranslation1d_t>();
   if (planar_joints.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto planar_joint_entities = allocator_.getQueryEntities(planar_joint_query_);

   bool render_all_pressed = false;
   bool hide_all_pressed = false;

   ImGui::PushID("translation 1d");
   if (planar_joint_entities.size() > 0)
   {
      render_all_pressed = ImGui::Button("Render all");
      hide_all_pressed = ImGui::Button("Hide all");
   }
   ImGui::PopID();

   for (const auto planar_joint_entity : planar_joint_entities)
   {
      std::string planar_joint_label("1d translation constraint ");
      planar_joint_label += std::to_string(planar_joint_entity);

      oy::types::constraintTranslation1d_t & planar_joint = *planar_joints[planar_joint_entity];
      trecs::edge_t & edge = *edges[planar_joint_entity];

      auto & viz_ids = *allocator_.getComponent<planar_joint_viz_ids_t>(planar_joint_entity);

      if (render_all_pressed)
      {
         viz_ids.render = true;
         renderer_.enableMesh(viz_ids.childLinkPointMeshId);
         renderer_.enableMesh(viz_ids.parentAxisMeshId);
         renderer_.enableMesh(viz_ids.parentLinkPointMeshId);
      }
      if (hide_all_pressed)
      {
         viz_ids.render = false;
         renderer_.disableMesh(viz_ids.childLinkPointMeshId);
         renderer_.disableMesh(viz_ids.parentAxisMeshId);
         renderer_.disableMesh(viz_ids.parentLinkPointMeshId);
      }

      geometry::types::isometricTransform_t trans_A_to_W = getTransform(
         edge.nodeIdA
      );

      geometry::types::isometricTransform_t trans_B_to_W = getTransform(
         edge.nodeIdB
      );

      // Force body B's link point to be the closest point on the constraint
      // plane to body B's center of mass position.

      const Vector3 parent_link_W = geometry::transform::forwardBound(
         trans_A_to_W, planar_joint.parentLinkPoint
      );

      const Vector3 parent_axis_W = geometry::transform::forwardUnbound(
         trans_A_to_W, planar_joint.parentAxis
      );

      const Vector3 child_link_W = geometry::transform::forwardBound(
         trans_B_to_W, planar_joint.childLinkPoint
      );

      Matrix33 R_axis_up = makeVectorUp(parent_axis_W);

      renderer_.updateMeshTransform(
         viz_ids.parentAxisMeshId,
         parent_link_W,
         R_axis_up.transpose(),
         identityMatrix()
      );

      renderer_.updateMeshTransform(
         viz_ids.parentLinkPointMeshId,
         parent_link_W,
         identityMatrix(),
         identityMatrix()
      );

      renderer_.updateMeshTransform(
         viz_ids.childLinkPointMeshId,
         child_link_W,
         identityMatrix(),
         identityMatrix()
      );

      translation1dUi(
         planar_joint_entity,
         planar_joint_label,
         edge,
         planar_joint,
         viz_ids
      );
   }
}

trecs::uid_t Translation1dConstraintWidget::addTranslation1d(
   const oy::types::bodyLink_t body_link,
   const oy::types::constraintTranslation1d_t & planar_joint
)
{
   trecs::uid_t new_planar_joint_uid = allocator_.addEntity(body_link.parentId, body_link.childId);
   allocator_.addComponent(new_planar_joint_uid, planar_joint);

   planar_joint_viz_ids_t viz_ids;
   viz_ids.render = true;

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );

   viz_ids.parentLinkPointMeshId = renderer_.addMesh(
      temp_mesh_data, lavender, 0
   );
   viz_ids.childLinkPointMeshId = renderer_.addMesh(
      temp_mesh_data, lavender, 0
   );

   geometry::types::triangleMesh_t cylinder = geometry::mesh::loadDefaultShapeMesh(
      geometry::types::enumShape_t::CYLINDER
   );
   geometry::mesh::scaleMesh(Vector3(0.05f, 0.05f, 5.f), cylinder);

   geometry::converters::to_pod(cylinder, &temp_mesh_data);
   viz_ids.parentAxisMeshId = renderer_.addMesh(
      temp_mesh_data, red, 0
   );

   allocator_.addComponent(new_planar_joint_uid, viz_ids);

   return new_planar_joint_uid;
}

void Translation1dConstraintWidget::translation1dUi(
   const trecs::uid_t planar_joint_uid,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::constraintTranslation1d_t & planar_joint,
   planar_joint_viz_ids_t & viz_ids
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
         ImGui::Text("Parent link point");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##1", &planar_joint.parentLinkPoint[0], 0.1f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Parent axis unit vector");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##2", &planar_joint.parentAxis[0], 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Child link point");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##3", &planar_joint.childLinkPoint[0], 0.1f);
         ImGui::PopItemWidth();

         ImGui::EndTable();
      }

      if (render_changed)
      {
         if (viz_ids.render)
         {
            renderer_.enableMesh(viz_ids.parentLinkPointMeshId);
            renderer_.enableMesh(viz_ids.childLinkPointMeshId);
            renderer_.enableMesh(viz_ids.parentAxisMeshId);
         }
         else
         {
            renderer_.disableMesh(viz_ids.parentLinkPointMeshId);
            renderer_.disableMesh(viz_ids.childLinkPointMeshId);
            renderer_.disableMesh(viz_ids.parentAxisMeshId);
         }
      }

      if (ImGui::Button("Delete planar joint"))
      {
         deleteComponent(planar_joint_uid);
      }

      ImGui::TreePop();
   }
}
