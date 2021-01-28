#include "editor_gui.hpp"
#include "misc/cpp/imgui_stdlib.h"

#include "data_model_io.h"
#include "logger_utils.hpp"

#include "slingshot_type_converters.hpp"

#include <algorithm>

bool EditorGui::no_window(void)
{
   if (show_open_dialog_)
   {
      ImGui::Begin("Open filename", &show_open_dialog_);

      ImGui::InputText("Filename", &open_name_);

      if (ImGui::Button("Load file"))
      {
         bool success = importScenario(open_name_);
         if (success)
         {
            show_open_dialog_ = false;
         }
      }

      ImGui::End();
   }

   if (ImGui::BeginMenuBar())
   {
      if (ImGui::BeginMenu("Menu"))
      {
         if (ImGui::MenuItem("New", nullptr))
         {
            clearEditor();
         }
         if (ImGui::MenuItem("Import", nullptr, &show_open_dialog_))
         {
         }
         if (ImGui::MenuItem("Save As"))
         {
         }
         ImGui::EndMenu();
      }
      ImGui::EndMenuBar();
   }

   if (ImGui::Button("Add body"))
   {
      trecs::uid_t new_body_entity = rigid_body_widget_.addDefaultComponent();

      oy::types::forceConstant_t gravity;
      gravity.acceleration.Initialize(0.f, 0.f, -9.8f);
      gravity.forceFrame = oy::types::enumFrame_t::GLOBAL;
      gravity.childLinkPoint.Initialize(0.f, 0.f, 0.f);

      constant_force_widget_.addConstantForce({-1, new_body_entity}, gravity);

      oy::types::torqueDrag_t drag_torque;
      drag_torque.linearDragCoeff = -0.04f;
      drag_torque.quadraticDragCoeff = 0.f;

      drag_torque_widget_.addDragTorque({-1, new_body_entity}, drag_torque);

      oy::types::forceDrag_t drag_force;
      drag_force.linearDragCoeff = -0.04f;
      drag_force.quadraticDragCoeff = 0.f;

      drag_force_widget_.addDragForce({-1, new_body_entity}, drag_force);
   }

   if (ImGui::Button("Add balljoint"))
   {
      balljoint_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add gear"))
   {
      gear_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add revolute joint"))
   {
      revolute_joint_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add revolute motor"))
   {
      revolute_motor_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add 1D rotation constraint"))
   {
      rotation_1d_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add 1D translation constraint"))
   {
      planar_joint_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add spring"))
   {
      spring_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add velocity damper"))
   {
      damper_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add constant force"))
   {
      constant_force_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add drag torque"))
   {
      drag_torque_widget_.addDefaultComponent();
   }

   if (ImGui::Button("Add drag force"))
   {
      drag_force_widget_.addDefaultComponent();
   }

   ImGui::InputText("Scenario name", &save_name_);

   if (ImGui::Button("Save scenario"))
   {
      saveScenario(save_name_.c_str());
   }

   rigid_body_widget_.componentsUi();
   balljoint_widget_.componentsUi();
   gear_widget_.componentsUi();
   revolute_joint_widget_.componentsUi();
   revolute_motor_widget_.componentsUi();
   rotation_1d_widget_.componentsUi();
   planar_joint_widget_.componentsUi();

   spring_widget_.componentsUi();
   damper_widget_.componentsUi();
   constant_force_widget_.componentsUi();

   drag_torque_widget_.componentsUi();
   drag_force_widget_.componentsUi();

   // uhhh.... ok?
   return false;
}

void EditorGui::clearEditor(void)
{
   auto entities = allocator_.getEntities();
   for (const auto entity : entities)
   {
      allocator_.removeEntity(entity);
   }

   renderer_.clear();
}

void EditorGui::saveAllocatorToScenario(oy::types::scenario_t & scenario) const
{
   scenario.clear();

   const auto rigid_body_entities = allocator_.getQueryEntities(rigid_body_query_);

   std::cout << "Saving " << rigid_body_entities.size() << " entities from the allocator to scenario\n";

   for (const auto body_id : rigid_body_entities)
   {
      scenario.isometric_colliders[body_id] = *allocator_.getComponent<oy::types::isometricCollider_t>(body_id);
      scenario.bodies[body_id] = *allocator_.getComponent<oy::types::rigidBody_t>(body_id);
      scenario.shapes[body_id] = *allocator_.getComponent<geometry::types::shape_t>(body_id);
      scenario.body_types[body_id] = allocator_.hasComponent<oy::types::StationaryBody>(body_id) ? oy::types::enumRigidBody_t::STATIONARY : oy::types::enumRigidBody_t::DYNAMIC;

      geometry::types::shape_t * temp_shape = allocator_.getComponent<geometry::types::shape_t>(body_id);
      if (temp_shape != nullptr)
      {
         scenario.shapes[body_id] = *temp_shape;
      }
   }

   addComponentsToScenario<oy::types::constraintBalljoint_t>(scenario.balljoints);
   addComponentsToScenario<oy::types::constraintGear_t>(scenario.gears);
   addComponentsToScenario<oy::types::constraintRevoluteJoint_t>(scenario.revolute_joints);
   addComponentsToScenario<oy::types::constraintRevoluteMotor_t>(scenario.revolute_motors);
   addComponentsToScenario<oy::types::constraintRotation1d_t>(scenario.rotation_1d);
   addComponentsToScenario<oy::types::constraintTranslation1d_t>(scenario.translation_1d);
   addComponentsToScenario<oy::types::forceSpring_t>(scenario.springs);
   addComponentsToScenario<oy::types::forceVelocityDamper_t>(scenario.dampers);
   addComponentsToScenario<oy::types::forceDrag_t>(scenario.drag_forces);
   addComponentsToScenario<oy::types::torqueDrag_t>(scenario.drag_torques);
   addComponentsToScenario<oy::types::forceConstant_t>(scenario.constant_forces);
}

void EditorGui::saveScenario(const char * save_name) const
{
   oy::types::scenario_t scenario_out;
   saveAllocatorToScenario(scenario_out);

   data_scenario_t scenario_data_out;
   oy::converters::to_pod(scenario_out, &scenario_data_out);

   std::string scenario_time_date_save = logger::utils::timeDateFilename(
      save_name, "scenario", "json"
   );

   write_data_to_file(&scenario_data_out, "data_scenario_t", scenario_time_date_save.c_str());

   clear_scenario(&scenario_data_out);

   auto rigid_body_entities = allocator_.getQueryEntities(rigid_body_query_);
   auto colors = allocator_.getComponents<viz::color_t>();

   data_vizConfig_t viz_config_data_out;
   viz_config_data_out.maxFps = 60;
   viz_config_data_out.cameraPoint.v[0] = 0.;
   viz_config_data_out.cameraPoint.v[1] = 0.;
   viz_config_data_out.cameraPoint.v[2] = 0.;

   viz_config_data_out.cameraPos.v[0] = -1.f;
   viz_config_data_out.cameraPos.v[1] = 0.f;
   viz_config_data_out.cameraPos.v[2] = 0.f;

   viz_config_data_out.mousePick = 0;
   viz_config_data_out.realtime = 1;
   viz_config_data_out.windowHeight = 600;
   viz_config_data_out.windowWidth = 800;

   viz_config_data_out.numMeshProps = rigid_body_entities.size();
   viz_config_data_out.meshProps = new data_vizMeshProperties_t[viz_config_data_out.numMeshProps];

   unsigned int i = 0;

   for (const auto rigid_body_entity : rigid_body_entities)
   {
      viz::color_t & body_color = *colors[rigid_body_entity];

      viz_config_data_out.meshProps[i].bodyId = static_cast<unsigned int>(rigid_body_entity);
      for (unsigned int j = 0; j < 4; ++j)
      {
         viz_config_data_out.meshProps[i].color.v[j] = body_color[j];
      }
      ++i;
   }

   std::string viz_config_time_date_save = logger::utils::timeDateFilename(
      save_name, "viz", "json"
   );

   write_data_to_file(
      &viz_config_data_out, "data_vizConfig_t", viz_config_time_date_save.c_str()
   );
}

bool EditorGui::importScenario(const std::string & scenario_filename)
{
   data_scenario_t scenario_data_in;
   initialize_scenario(&scenario_data_in);

   int success = read_data_from_file(
      &scenario_data_in, scenario_filename.c_str()
   );

   if (!success)
   {
      clear_scenario(&scenario_data_in);
      return false;
   }

   oy::types::scenario_t scenario_in;
   oy::converters::from_pod(&scenario_data_in, scenario_in);

   std::map<int, int> scen_to_editor_id;
   scen_to_editor_id[-1] = -1;

   for (
      auto body_it = scenario_in.bodies.begin();
      body_it != scenario_in.bodies.end();
      ++body_it
   )
   {
      auto collider_it = scenario_in.isometric_colliders.find(body_it->first);

      if (collider_it == scenario_in.isometric_colliders.end())
      {
         std::cout << "Couldn't find collider for body id: " << body_it->first << "\n";
         continue;
      }

      auto shape_it = scenario_in.shapes.find(body_it->first);

      if (shape_it == scenario_in.shapes.end())
      {
         std::cout << "Couldn't find shape for body id: " << body_it->first << "\n";
         continue;
      }

      auto body_type_it = scenario_in.body_types.find(body_it->first);

      if (body_type_it == scenario_in.body_types.end())
      {
         std::cout << "Couldn't stationary/dynamic tag for body id: " << body_it->first << "\n";
         continue;
      }

      int new_body_uid = rigid_body_widget_.addRigidBody(
         body_it->second,
         collider_it->second,
         shape_it->second,
         (body_type_it->second == oy::types::enumRigidBody_t::STATIONARY)
      );

      scen_to_editor_id[body_it->first] = new_body_uid;
   }

   addComponentsFromScenario(scenario_in.balljoints, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.gears, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.revolute_joints, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.revolute_motors, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.rotation_1d, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.translation_1d, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.springs, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.dampers, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.drag_forces, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.drag_torques, scen_to_editor_id);
   addComponentsFromScenario(scenario_in.constant_forces, scen_to_editor_id);

   return true;
}
