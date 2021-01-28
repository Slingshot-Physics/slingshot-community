#include "gear_widget.hpp"

#include "attitudeutils.hpp"
#include "geometry.hpp"
#include "geometry_type_converters.hpp"

#include "imgui.h"

trecs::uid_t GearConstraintWidget::addDefaultComponent(void)
{
   oy::types::constraintGear_t default_gear;
   default_gear.parentAxis[2] = 1.f;
   default_gear.childAxis[2] = 1.f;
   default_gear.parentGearRadius = 1.f;
   default_gear.childGearRadius = 1.f;
   default_gear.rotateParallel = false;

   return addGear({-1, -1}, default_gear);
}

void GearConstraintWidget::deleteComponent(trecs::uid_t entity)
{
   auto viz_ids = allocator_.getComponent<gear_viz_ids_t>(entity);
   renderer_.deleteRenderable(viz_ids->axisParentMeshId);
   renderer_.deleteRenderable(viz_ids->axisChildMeshId);
   allocator_.removeEntity(entity);
}

void GearConstraintWidget::componentsUi(void)
{
   auto gears = allocator_.getComponents<oy::types::constraintGear_t>();
   if (gears.empty())
   {
      return;
   }

   auto edges = allocator_.getComponents<trecs::edge_t>();

   const auto gear_entities = allocator_.getQueryEntities(gear_query_);

   for (const auto gear_entity : gear_entities)
   {
      std::string gear_label("gear ");
      gear_label += std::to_string(gear_entity);

      oy::types::constraintGear_t & gear = *gears[gear_entity];
      trecs::edge_t & edge = *edges[gear_entity];

      gear_viz_ids_t & viz_ids = *allocator_.getComponent<gear_viz_ids_t>(gear_entity);

      geometry::types::isometricTransform_t trans_A_to_W = getTransform(
         edge.nodeIdA
      );

      Vector3 axis_a_W = trans_A_to_W.rotate * gear.parentAxis;
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

      Vector3 axis_b_W = trans_B_to_W.rotate * gear.childAxis;
      Matrix33 R_W_to_axis_B = makeVectorUp(axis_b_W);

      renderer_.updateMeshTransform(
         viz_ids.axisChildMeshId,
         trans_B_to_W.translate,
         R_W_to_axis_B.transpose(),
         identityMatrix()
      );

      gearUi(
         gear_entity,
         gear_label,
         edge,
         gear,
         viz_ids
      );
   }
}

trecs::uid_t GearConstraintWidget::addGear(
   const oy::types::bodyLink_t body_link,
   const oy::types::constraintGear_t & gear
)
{
   trecs::uid_t new_gear_uid = allocator_.addEntity(body_link.parentId, body_link.childId);
   allocator_.addComponent(new_gear_uid, gear);

   gear_viz_ids_t viz_ids;
   viz_ids.render = true;

   geometry::types::triangleMesh_t cylinder = geometry::mesh::loadDefaultShapeMesh(
      geometry::types::enumShape_t::CYLINDER
   );
   geometry::mesh::scaleMesh(Vector3(0.05f, 0.05f, 5.f), cylinder);
   data_triangleMesh_t cylinder_data;
   geometry::converters::to_pod(cylinder, &cylinder_data);

   viz_ids.axisParentMeshId = renderer_.addMesh(cylinder_data, red, 0);
   viz_ids.axisChildMeshId = renderer_.addMesh(cylinder_data, green, 0);

   allocator_.addComponent(new_gear_uid, viz_ids);

   return new_gear_uid;
}

void GearConstraintWidget::gearUi(
   const trecs::uid_t gear_uid,
   const std::string & label,
   trecs::edge_t & edge,
   oy::types::constraintGear_t & gear,
   gear_viz_ids_t & viz_ids
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
         ImGui::Text("Rotate parallel");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::Checkbox("##3", &gear.rotateParallel);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Gear radius A");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##1", &gear.parentGearRadius, 0.1f, 0.f, 100.f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Gear radius B");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##2", &gear.childGearRadius, 0.1f, 0.f, 100.f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Axis body A");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##4", &gear.parentAxis[0], 0.01f);
         gear.parentAxis.Normalize();
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("Axis body B");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##5", &gear.childAxis[0], 0.01f);
         gear.childAxis.Normalize();
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

      if (ImGui::Button("Delete gear"))
      {
         deleteComponent(gear_uid);
      }

      ImGui::TreePop();
   }
}
