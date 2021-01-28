#include <iostream>

#include "data_model.h"
#include "slingshot_type_converters.hpp"
#include "logger_utils.hpp"
#include "vector3.hpp"

#include "line_loader_ui.hpp"

void LineLoader::operator()(void)
{
   ImGui::Begin("Geometry manipulatrix");

   no_window();

   ImGui::End();
}

bool LineLoader::no_window(void)
{
   bool ui_modified = false;
   ImGui::Text("%s", prefix_.c_str());

   if (
      ImGui::BeginCombo(
         combo_name_.c_str(), line_type_names_[line_type].c_str()
      )
   )
   {
      for (int i = 0; i < 3; ++i)
      {
         bool is_selected = (line_type == i);
         if (ImGui::Selectable(line_type_names_[i].c_str(), is_selected))
         {
            line_type = i;
         }
         if (is_selected)
         {
            ImGui::SetItemDefaultFocus();
         }
      }

      ImGui::EndCombo();
   }

   ui_modified |= ImGui::DragFloat3(
      start_point_name_.c_str(), &start[0], 0.025f, -15.f, 15.f, "%0.5f", 1.f
   );

   ui_modified |= ImGui::DragFloat3(
      slope_name_.c_str(), &slope[0], 0.025f, -1.f, 1.f, "%0.5f", 1.f
   );

   if (line_type == 2)
   {
      ui_modified |= ImGui::DragFloat(
         length_name_.c_str(), &length, 1e-1f, 0.f, 50.f, "%0.5f", 1.f
      );
   }

   slope.Normalize();

   update_line_points();

   ImGui::Text("End point: %f, %f, %f", line_points[1][0], line_points[1][1], line_points[1][2]);

   return ui_modified;
}

void LineLoader::update_line_points(void)
{
   if (line_type == 2)
   {
      line_points[0] = start;
      line_points[1] = start + slope * length;
   }
   else if (line_type == 1)
   {
      line_points[0] = start;
      line_points[1] = start + slope * 100.f;
   }
   else if (line_type == 0)
   {
      line_points[0] = start - slope * 100.f;
      line_points[1] = start + slope * 100.f;
   }
}

void LineLoader::prepend_prefix(std::string & name)
{
   name.insert(0, " ");
   name.insert(0, prefix_);
}
