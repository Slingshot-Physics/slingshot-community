#include "rigid_body_widget.hpp"

#include "attitudeutils.hpp"
#include "geometry.hpp"
#include "geometry_type_converters.hpp"
#include "shape_widgets.hpp"

#include "imgui.h"

void shapeComboBox(
   const std::map<geometry::types::enumShape_t, std::string> & shape_type_to_names,
   geometry::types::enumShape_t & current_shape,
   const char * combo_box_label
)
{
   if (ImGui::BeginCombo(combo_box_label, shape_type_to_names.at(current_shape).c_str()))
   {
      for (const auto & type_to_name : shape_type_to_names)
      {
         const bool is_selected = (current_shape == type_to_name.first);

         if (ImGui::Selectable(type_to_name.second.c_str(), is_selected))
         {
            current_shape = type_to_name.first;
         }

         if (is_selected)
         {
            ImGui::SetItemDefaultFocus();
         }
      }

      ImGui::EndCombo();
   }
}

bool shapeUi(geometry::types::shape_t & shape)
{
   bool shape_modified = false;

   switch(shape.shapeType)
   {
      case geometry::types::enumShape_t::CUBE:
      {
         shape_modified |= cube_widget(shape);
         break;
      }
      case geometry::types::enumShape_t::SPHERE:
      {
         shape_modified |= sphere_widget(shape);
         break;
      }
      case geometry::types::enumShape_t::CAPSULE:
      {
         shape_modified |= capsule_widget(shape);
         break;
      }
      case geometry::types::enumShape_t::CYLINDER:
      {
         shape_modified |= cylinder_widget(shape);
         break;
      }
      default:
         break;
   }

   return shape_modified;
}

trecs::uid_t RigidBodyWidget::addDefaultComponent(void)
{
   oy::types::rigidBody_t default_body;
   default_body.linPos.Initialize(0.f, 0.f, 0.f);
   default_body.linVel.Initialize(0.f, 0.f, 0.f);
   default_body.inertiaTensor = identityMatrix();
   default_body.mass = 1.f;
   default_body.ql2b.Initialize(1.f, 0.f, 0.f, 0.f);

   oy::types::isometricCollider_t default_collider;
   default_collider.enabled = 1;
   default_collider.mu = 0.5f;
   default_collider.restitution = 0.5f;

   geometry::types::shape_t default_shape = geometry::mesh::defaultShape(
      geometry::types::enumShape_t::CUBE
   );

   trecs::uid_t new_body_entity = addRigidBody(
      default_body, default_collider, default_shape, false
   );

   return new_body_entity;
}

trecs::uid_t RigidBodyWidget::addRigidBody(
   const oy::types::rigidBody_t & body,
   const oy::types::isometricCollider_t & collider,
   const geometry::types::shape_t & shape,
   const bool stationary
)
{
   trecs::uid_t new_body_uid = allocator_.addEntity();

   allocator_.addComponent(new_body_uid, body);
   allocator_.addComponent(new_body_uid, collider);
   allocator_.addComponent(new_body_uid, shape);

   if (!stationary)
   {
      allocator_.addComponent(new_body_uid, oy::types::DynamicBody{});
   }
   else
   {
      allocator_.addComponent(new_body_uid, oy::types::StationaryBody{});
   }

   geometry::types::triangleMesh_t shape_mesh = geometry::mesh::loadShapeMesh(shape);
   data_triangleMesh_t shape_mesh_data;

   geometry::converters::to_pod(shape_mesh, &shape_mesh_data);

   body_viz_id_t viz_id;
   viz_id.meshId = renderer_.addMesh(shape_mesh_data, gray, 0);
   viz_id.render = true;

   allocator_.addComponent(new_body_uid, viz_id);
   allocator_.addComponent(new_body_uid, gray);

   const float rad2deg = 180.f / M_PI;

   rpy_deg_t rpy_vec;
   rpy_vec.rpyDeg = quaternionToAttitude(body.ql2b.conjugate()) * rad2deg;
   allocator_.addComponent(new_body_uid, rpy_vec);

   return new_body_uid;
}

void RigidBodyWidget::deleteComponent(trecs::uid_t entity)
{
   auto rigid_body_viz = allocator_.getComponent<body_viz_id_t>(entity);
   renderer_.deleteRenderable(rigid_body_viz->meshId);
   allocator_.removeEntity(entity);
}

void RigidBodyWidget::componentsUi(void)
{
   auto rigid_body_entities = allocator_.getQueryEntities(rigid_body_query_);
   for (const auto entity : rigid_body_entities)
   {
      std::string label("body ");
      label += std::to_string(entity);

      oy::types::rigidBody_t * temp_body = allocator_.getComponent<oy::types::rigidBody_t>(entity);
      oy::types::isometricCollider_t * temp_collider = allocator_.getComponent<oy::types::isometricCollider_t>(entity);
      geometry::types::shape_t * temp_shape = allocator_.getComponent<geometry::types::shape_t>(entity);
      viz::color_t * body_color = allocator_.getComponent<viz::color_t>(entity);
      body_viz_id_t * viz_id = allocator_.getComponent<body_viz_id_t>(entity);
      rpy_deg_t * rpy_deg = allocator_.getComponent<rpy_deg_t>(entity);

      rigidBodyUi(
         entity,
         *temp_body,
         *temp_collider,
         *temp_shape,
         *rpy_deg,
         *viz_id,
         *body_color,
         label
      );
   }
}

void RigidBodyWidget::rigidBodyUi(
   trecs::uid_t body_entity,
   oy::types::rigidBody_t & body,
   oy::types::isometricCollider_t & collider,
   geometry::types::shape_t & shape,
   rpy_deg_t & rpy_deg,
   body_viz_id_t & viz_id,
   viz::color_t & color,
   const std::string & label
)
{
   ImGui::PushStyleColor(
      ImGuiCol_Text,
      ImVec4(color[0], color[1], color[2], color[3])
   );

   const float deg2rad = M_PI / 180.f;

   if (ImGui::TreeNode(label.c_str()))
   {
      ImGui::PopStyleColor();
      geometry::types::enumShape_t start_shape = shape.shapeType;

      if (ImGui::ColorEdit4("color", &(color[0])))
      {
         renderer_.updateMeshColor(viz_id.meshId, color);
      }

      shapeComboBox(
         shape_type_to_name_, shape.shapeType, "shapes"
      );

      if (shape.shapeType != start_shape)
      {
         data_triangleMesh_t temp_mesh_data = geometry::mesh::loadDefaultShapeMeshData(shape.shapeType, 1.f);
         renderer_.updateMesh(viz_id.meshId, temp_mesh_data);

         shape = geometry::mesh::defaultShape(shape.shapeType);
      }

      if (ImGui::BeginTable("fields", 2, ImGuiTableFlags_Borders))
      {
         ImGui::TableSetupColumn("Field name", ImGuiTableColumnFlags_WidthFixed);
         ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthStretch);
         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("position");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("pos", &body.linPos[0], 0.1f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("roll/pitch/yaw");
         ImGui::TableSetColumnIndex(1);

         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("rpy", &rpy_deg.rpyDeg[0], 0.1f);
         ImGui::PopItemWidth();

         body.ql2b = attitudeToQuaternion(rpy_deg.rpyDeg * deg2rad).conjugate();

         bool shape_modified = shapeUi(shape);

         if (shape_modified)
         {
            geometry::types::triangleMesh_t temp_mesh = geometry::mesh::loadShapeMesh(shape);
            data_triangleMesh_t temp_mesh_data;
            geometry::converters::to_pod(temp_mesh, &temp_mesh_data);
            renderer_.updateMesh(viz_id.meshId, temp_mesh_data);
         }

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("velocity");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##velocity", &(body.linVel[0]), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("angular velocity");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##angular_velocity", &(body.angVel[0]), 0.01f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("mass");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat("##2mass", &body.mass, 0.01f, 0.0001f, 1024.f);
         ImGui::PopItemWidth();

         Vector3 inertia_diag = body.mass * inertia(shape);

         for (unsigned int i = 0; i < 3; ++i)
         {
            body.inertiaTensor(i, i) = inertia_diag[i];
         }

         // For viewing only
         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("inertia diagonal");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::DragFloat3("##inertia_diagonal", &(inertia_diag[0]), 0.01f, 0.0005f, 64.f, "%0.5f");
         ImGui::PopItemWidth();

         bool body_stationary = allocator_.hasComponent<oy::types::StationaryBody>(body_entity);
         bool originally_stationary = body_stationary;
         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("stationary");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         bool stationarity_changed = ImGui::Checkbox("stationary", &body_stationary);
         ImGui::PopItemWidth();
         if (stationarity_changed)
         {
            if (originally_stationary)
            {
               allocator_.removeComponent<oy::types::StationaryBody>(body_entity);
               allocator_.addComponent(body_entity, oy::types::DynamicBody{});
            }
            else
            {
               allocator_.removeComponent<oy::types::DynamicBody>(body_entity);
               allocator_.addComponent(body_entity, oy::types::StationaryBody{});
            }
         }

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("collider");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::Checkbox("enabled", &collider.enabled);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("friction");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::SliderFloat("coefficient of friction", &collider.mu, 0.f, 1.f);
         ImGui::PopItemWidth();

         ImGui::TableNextRow();
         ImGui::TableSetColumnIndex(0);
         ImGui::Text("restitution");
         ImGui::TableSetColumnIndex(1);
         ImGui::PushItemWidth(-1);
         ImGui::SliderFloat("coefficient of restitution", &collider.restitution, 0.01f, 1.f);
         ImGui::PopItemWidth();

         ImGui::EndTable();
      }
      if (ImGui::Button("delete"))
      {
         deleteComponent(body_entity);
      }

      ImGui::TreePop();
   }
   else
   {
      ImGui::PopStyleColor();
   }

   Quaternion q_b2l = ~(body.ql2b);
   Matrix33 R_b2l = q_b2l.rotationMatrix();

   geometry::types::transform_t trans_C_to_W;
   trans_C_to_W.translate = body.linPos;
   trans_C_to_W.rotate = R_b2l;
   trans_C_to_W.scale = identityMatrix();

   renderer_.updateMeshTransform(
      viz_id.meshId,
      trans_C_to_W.translate,
      trans_C_to_W.rotate,
      trans_C_to_W.scale
   );
}
