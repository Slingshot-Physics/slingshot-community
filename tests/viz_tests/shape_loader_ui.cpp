#include "shape_loader_ui.hpp"

#include <algorithm>

int ShapeLoader::counter_ = 0;

bool ShapeLoader::no_window(void)
{
   bool ui_modified = false;

   ImGui::SetNextItemOpen(true);
   if (ImGui::TreeNode(name_.c_str()))
   {
      int old_selection = selected_shape_type_id_;

      std::string combo_preview = shape_enums_to_names_[
         int_to_shape(selected_shape_type_id_)
      ];

      if (ImGui::BeginCombo("shape", combo_preview.c_str()))
      {
         for (auto & shape_combo: shape_enums_to_names_)
         {
            bool shape_selected = (
               selected_shape_type_id_ == shape_to_int(shape_combo.first)
            );

            if (ImGui::Selectable(shape_combo.second.c_str(), shape_selected))
            {
               selected_shape_type_id_ = shape_to_int(shape_combo.first);
            }

            if (shape_selected)
            {
               ImGui::SetItemDefaultFocus();
            }
         }

         ImGui::EndCombo();
      }

      shape_.shapeType = int_to_shape(selected_shape_type_id_);

      switch(shape_.shapeType)
      {
         case geometry::types::enumShape_t::CAPSULE:
            ui_modified |= capsule_widget();
            break;
         case geometry::types::enumShape_t::CYLINDER:
            ui_modified |= cylinder_widget();
            break;
         case geometry::types::enumShape_t::SPHERE:
            ui_modified |= sphere_widget();
            break;
         case geometry::types::enumShape_t::CUBE:
            ui_modified |= cube_widget();
            break;
         default:
            break;
      }

      bool transform_modified = transform_ui_.no_window();

      ui_modified |= (
         transform_modified ||
         old_selection != selected_shape_type_id_
      );

      if (ui_modified)
      {
         mesh_ = geometry::mesh::loadShapeMesh(shape_);
      }

      ImGui::TreePop();
   }

   return ui_modified;
}

bool ShapeLoader::capsule_widget(void)
{
   bool widget_modified = false;
   ImGui::Separator();
   ImGui::Text("Capsule");
   widget_modified |= ImGui::DragFloat("radius", &shape_.capsule.radius, 0.1f, 0.1f, 15.f);
   widget_modified |= ImGui::DragFloat("height", &shape_.capsule.height, 0.1f, 0.1f, 15.f);
   shape_.capsule.height = std::max(shape_.capsule.radius, shape_.capsule.height);

   return widget_modified;
}

bool ShapeLoader::cylinder_widget(void)
{
   bool widget_modified = false;
   ImGui::Separator();
   ImGui::Text("Cylinder");
   widget_modified |= ImGui::DragFloat("radius", &shape_.cylinder.radius, 0.1f, 0.1f, 15.f);
   widget_modified |= ImGui::DragFloat("height", &shape_.cylinder.height, 0.1f, 0.1f, 15.f);

   return widget_modified;
}

bool ShapeLoader::sphere_widget(void)
{
   bool widget_modified = false;
   ImGui::Separator();
   ImGui::Text("Sphere");
   widget_modified |= ImGui::DragFloat("radius", &shape_.sphere.radius, 0.1f, 0.1f, 15.f);

   return widget_modified;
}

bool ShapeLoader::cube_widget(void)
{
   bool widget_modified = false;
   ImGui::Separator();
   ImGui::Text("Cube");
   widget_modified |= ImGui::DragFloat("length", &shape_.cube.length, 0.1f, 0.1f, 15.f);
   widget_modified |= ImGui::DragFloat("width", &shape_.cube.width, 0.1f, 0.1f, 15.f);
   widget_modified |= ImGui::DragFloat("height", &shape_.cube.height, 0.1f, 0.1f, 15.f);

   shape_.cube.length = std::max(shape_.cube.length, 0.1f);
   shape_.cube.width = std::max(shape_.cube.width, 0.1f);
   shape_.cube.height = std::max(shape_.cube.height, 0.1f);

   return widget_modified;
}
