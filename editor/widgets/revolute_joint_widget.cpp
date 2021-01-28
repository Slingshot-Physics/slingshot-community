#include "revolute_joint_widget.hpp"

#include "geometry.hpp"

#include "imgui.h"

trecs::uid_t RevoluteJointConstraintWidget::addDefaultComponent(void)
{
   oy::types::constraintRevoluteJoint_t default_revolute_joint;
   default_revolute_joint.parentLinkPoints[0][2] = 1.f;
   default_revolute_joint.parentLinkPoints[1][2] = -1.f;

   default_revolute_joint.childLinkPoints[0][2] = 1.f;
   default_revolute_joint.childLinkPoints[1][2] = -1.f;

   return addRevoluteJoint({-1, -1}, default_revolute_joint);
}

void RevoluteJointConstraintWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<revolute_joint_viz_ids_t>(entity);
   renderer_.deleteRenderable(viz_ids->linkPointParentMeshIds[0]);
   renderer_.deleteRenderable(viz_ids->linkPointParentMeshIds[1]);
   renderer_.deleteRenderable(viz_ids->linkPointChildMeshIds[0]);
   renderer_.deleteRenderable(viz_ids->linkPointChildMeshIds[1]);
   allocator_.removeEntity(entity);
}

void RevoluteJointConstraintWidget::componentsUi(void)
{
   auto revolute_joints = allocator_.getComponents<oy::types::constraintRevoluteJoint_t>();
   if (revolute_joints.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto revolute_joint_entities = allocator_.getQueryEntities(revolute_joint_query_);

   for (const auto revolute_joint_entity : revolute_joint_entities)
   {
      std::string revolute_joint_label("revolute joint ");
      revolute_joint_label += std::to_string(revolute_joint_entity);

      oy::types::constraintRevoluteJoint_t & revolute_joint = *revolute_joints[revolute_joint_entity];
      trecs::edge_t & edge = *edges[revolute_joint_entity];

      auto & viz_ids = *allocator_.getComponent<revolute_joint_viz_ids_t>(revolute_joint_entity);

      geometry::types::isometricTransform_t trans_A_to_W = getTransform(
         edge.nodeIdA
      );

      geometry::types::isometricTransform_t trans_B_to_W = getTransform(
         edge.nodeIdB
      );

      for (unsigned int i = 0; i < 2; ++i)
      {
         Vector3 link_point_W = geometry::transform::forwardBound(
            trans_A_to_W, revolute_joint.parentLinkPoints[i]
         );

         revolute_joint.childLinkPoints[i] = geometry::transform::inverseBound(
            trans_B_to_W, link_point_W
         );

         renderer_.updateMeshTransform(
            viz_ids.linkPointParentMeshIds[i],
            geometry::transform::forwardBound(
               trans_A_to_W, revolute_joint.parentLinkPoints[i]
            ),
            identityMatrix(),
            identityMatrix()
         );

         renderer_.updateMeshTransform(
            viz_ids.linkPointChildMeshIds[i],
            geometry::transform::forwardBound(
               trans_B_to_W, revolute_joint.childLinkPoints[i]
            ),
            identityMatrix(),
            identityMatrix()
         );
      }

      revoluteJointUi(
         revolute_joint_entity,
         revolute_joint_label,
         edge,
         revolute_joint,
         viz_ids
      );
   }
}

trecs::uid_t RevoluteJointConstraintWidget::addRevoluteJoint(
   const oy::types::bodyLink_t body_link,
   const oy::types::constraintRevoluteJoint_t & revolute_joint
)
{
   trecs::uid_t new_revolute_joint_uid = allocator_.addEntity(body_link.parentId, body_link.childId);
   allocator_.addComponent(new_revolute_joint_uid, revolute_joint);

   revolute_joint_viz_ids_t viz_ids;
   viz_ids.render = true;

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );

   viz_ids.linkPointParentMeshIds[0] = renderer_.addMesh(
      temp_mesh_data, lavender, 0
   );
   viz_ids.linkPointParentMeshIds[1] = renderer_.addMesh(
      temp_mesh_data, lavender, 0
   );

   viz_ids.linkPointChildMeshIds[0] = renderer_.addMesh(
      temp_mesh_data, lavender, 0
   );
   viz_ids.linkPointChildMeshIds[1] = renderer_.addMesh(
      temp_mesh_data, lavender, 0
   );

   allocator_.addComponent(new_revolute_joint_uid, viz_ids);

   return new_revolute_joint_uid;
}

void RevoluteJointConstraintWidget::revoluteJointUi(
   const trecs::uid_t revolute_joint_uid,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::constraintRevoluteJoint_t & revolute_joint,
   revolute_joint_viz_ids_t & viz_ids
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

         for (unsigned int i = 0; i < 2; ++i)
         {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            char actual_label[32];
            snprintf(actual_label, 32, "parent link points %u", i);
            ImGui::Text("%s", actual_label);
            ImGui::TableSetColumnIndex(1);
            ImGui::PushItemWidth(-1);
            char label[32];
            snprintf(label, 32, "##%ulinkpoints", i);
            ImGui::DragFloat3(label, &(revolute_joint.parentLinkPoints[i][0]), 0.01f);
            ImGui::PopItemWidth();
         }

         ImGui::EndTable();
      }

      if (render_changed)
      {
         if (viz_ids.render)
         {
            for (unsigned int i = 0; i < 2; ++i)
            {
               renderer_.enableMesh(viz_ids.linkPointParentMeshIds[i]);
               renderer_.enableMesh(viz_ids.linkPointChildMeshIds[i]);
            }
         }
         else
         {
            for (unsigned int i = 0; i < 2; ++i)
            {
               renderer_.disableMesh(viz_ids.linkPointParentMeshIds[i]);
               renderer_.disableMesh(viz_ids.linkPointChildMeshIds[i]);
            }
         }
      }

      if (ImGui::Button("Delete revolute joint"))
      {
         deleteComponent(revolute_joint_uid);
      }

      ImGui::TreePop();
   }
}
