#ifndef EDITOR_GUI_HEADER
#define EDITOR_GUI_HEADER

#include "gui_callback_base.hpp"
#include "viz_renderer.hpp"

#include "allocator.hpp"

#include "widgets/rigid_body_widget.hpp"

#include "widgets/component_widget_interface.hpp"
#include "widgets/balljoint_widget.hpp"
#include "widgets/gear_widget.hpp"
#include "widgets/revolute_joint_widget.hpp"
#include "widgets/revolute_motor_widget.hpp"
#include "widgets/rotation_1d_widget.hpp"
#include "widgets/translation_1d_widget.hpp"

#include "widgets/constant_force_widget.hpp"
#include "widgets/damper_force_widget.hpp"
#include "widgets/drag_force_widget.hpp"
#include "widgets/drag_torque_widget.hpp"
#include "widgets/spring_force_widget.hpp"

#include "slingshot_types.hpp"
#include "geometry_types.hpp"

#include <map>
#include <vector>

class EditorGui : public viz::GuiCallbackBase
{
   public:

      const viz::color_t cyan = {41.f/255.f, 221.f/255.f, 244.f/255.f, 1.f};

      const viz::color_t lavender = {133.f/255.f, 75.f/255.f, 221.f/255.f, 1.f};

      const viz::color_t red = {1.f, 0.f, 0.f, 1.f};

      const viz::color_t green = {0.f, 1.f, 0.f, 1.f};

      const viz::color_t blue = {0.f, 0.f, 1.f, 1.f};

      const viz::color_t gray = {0.5f, 0.5f, 0.5f, 1.f};

      EditorGui(
         trecs::Allocator & allocator,
         viz::VizRenderer & renderer
      )
         : show_open_dialog_(false)
         , allocator_(allocator)
         , renderer_(renderer)
         , rigid_body_widget_(allocator, renderer)
         , balljoint_widget_(allocator, renderer)
         , gear_widget_(allocator, renderer)
         , revolute_joint_widget_(allocator, renderer)
         , revolute_motor_widget_(allocator, renderer)
         , rotation_1d_widget_(allocator, renderer)
         , planar_joint_widget_(allocator, renderer)
         , constant_force_widget_(allocator, renderer)
         , damper_widget_(allocator, renderer)
         , drag_force_widget_(allocator, renderer)
         , drag_torque_widget_(allocator, renderer)
         , spring_widget_(allocator, renderer)
      {
         rigid_body_query_ = allocator_.addArchetypeQuery<oy::types::rigidBody_t>();
      }

      void operator()(void)
      {
         ImGui::Begin("slingshot editor", nullptr, ImGuiWindowFlags_MenuBar);
         no_window();
         ImGui::End();
      }

      bool no_window(void);

   private:

      std::string save_name_;

      std::string open_name_;

      bool show_open_dialog_;

      trecs::query_t rigid_body_query_;

      const std::map<geometry::types::enumShape_t, std::string> shape_type_to_name_ = {
         {geometry::types::enumShape_t::CUBE, "cube"},
         {geometry::types::enumShape_t::CYLINDER, "cylinder"},
         {geometry::types::enumShape_t::SPHERE, "sphere"},
         {geometry::types::enumShape_t::CAPSULE, "capsule"},
      };

      trecs::Allocator & allocator_;

      viz::VizRenderer & renderer_;

      RigidBodyWidget rigid_body_widget_;

      BalljointConstraintWidget balljoint_widget_;

      GearConstraintWidget gear_widget_;

      RevoluteJointConstraintWidget revolute_joint_widget_;

      RevoluteMotorConstraintWidget revolute_motor_widget_;

      Rotation1dConstraintWidget rotation_1d_widget_;

      Translation1dConstraintWidget planar_joint_widget_;

      ConstantForceWidget constant_force_widget_;

      DamperForceWidget damper_widget_;

      DragForceWidget drag_force_widget_;

      DragTorqueWidget drag_torque_widget_;

      SpringForceWidget spring_widget_;

      // Clears out the allocator and renderer. Used for generating new
      // scenarios or for clearing out the editor before a scenario is loaded
      // from a file
      void clearEditor(void);

      // Converts the entire allocator state into a scenario data type.
      void saveAllocatorToScenario(oy::types::scenario_t & scenario) const;

      // Saves the editor state into a scenario data file under `save_name`.
      void saveScenario(const char * save_name) const;

      // Attempts to load a scenario from the given scenario_filename and
      // populate the editor with its data.
      bool importScenario(const std::string & scenario_filename);

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::constraintBalljoint_t & comp
      )
      {
         balljoint_widget_.addBalljoint(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::constraintGear_t & comp
      )
      {
         gear_widget_.addGear(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::constraintRevoluteJoint_t & comp
      )
      {
         revolute_joint_widget_.addRevoluteJoint(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::constraintRevoluteMotor_t & comp
      )
      {
         revolute_motor_widget_.addRevoluteMotor(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::constraintRotation1d_t & comp
      )
      {
         rotation_1d_widget_.addRotation1d(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::constraintTranslation1d_t & comp
      )
      {
         planar_joint_widget_.addTranslation1d(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::forceSpring_t & comp
      )
      {
         spring_widget_.addSpringForce(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::forceVelocityDamper_t & comp
      )
      {
         damper_widget_.addDamperForce(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::torqueDrag_t & comp
      )
      {
         drag_torque_widget_.addDragTorque(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::forceDrag_t & comp
      )
      {
         drag_force_widget_.addDragForce(body_link, comp);
      }

      void addGenericComponent(
         oy::types::bodyLink_t body_link,
         const oy::types::forceConstant_t & comp
      )
      {
         constant_force_widget_.addConstantForce(body_link, comp);
      }

      template <typename Component_T>
      void addComponentsToScenario(
         std::vector<std::pair<oy::types::bodyLink_t, Component_T> > & scenario_components
      ) const
      {
         const auto edges = allocator_.getComponents<trecs::edge_t>();
         const auto components = allocator_.getComponents<Component_T>();
         const auto entities = components.getUids();

         for (const auto & entity : entities)
         {
            oy::types::bodyLink_t body_link = {
               edges[entity]->nodeIdA, edges[entity]->nodeIdB
            };
            scenario_components.push_back(
               {body_link, *components[entity]}
            );
         }
      }

      // Takes a map of components (usually from a scenario type) and adds them
      // to the editor's state.
      // scen_to_fzx_ids is a map that converts the body IDs from the scenario
      // definition into body IDs used by the editor.
      template <typename Component_T>
      void addComponentsFromScenario(
         const std::vector<std::pair<oy::types::bodyLink_t, Component_T> > & components,
         const std::map<int, int> & scen_to_fzx_ids
      )
      {
         for (const auto & linked_component : components)
         {
            // Require that constraints and forces from scenario files refer to
            // body IDs that exist in the scenario itself.
            if (
               (scen_to_fzx_ids.find(linked_component.first.parentId) != scen_to_fzx_ids.end()) &&
               (scen_to_fzx_ids.find(linked_component.first.childId) != scen_to_fzx_ids.end()) &&
               (
                  (linked_component.first.parentId != -1) ||
                  (linked_component.first.childId != -1)
               )
            )
            {
               oy::types::bodyLink_t body_link;

               body_link.parentId = \
                  scen_to_fzx_ids.at(linked_component.first.parentId);
               body_link.childId = \
                  scen_to_fzx_ids.at(linked_component.first.childId);

               const Component_T temp_component = linked_component.second;

               addGenericComponent(body_link, temp_component);
            }
            else
            {
               std::cout << "Couldn't load a body-linked component with links: " << linked_component.first.parentId << ", " << linked_component.first.childId << "\n";
            }
         }
      }

};

#endif
