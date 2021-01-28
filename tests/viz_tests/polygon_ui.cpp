#include "polygon_ui.hpp"

#include "attitudeutils.hpp"
#include "data_vector3.h"
#include "geometry_type_converters.hpp"
#include "random_utils.hpp"

#include <random>

void PolygonUI::operator()(void)
{
   ImGui::Begin("Polygon UI");

   no_window();

   ImGui::End();
}

bool PolygonUI::no_window(void)
{
   char randomize_name[120];
   sprintf(randomize_name, "%s randomize!", prefix_.c_str());
   if (ImGui::Button(randomize_name))
   {
      num_points = std::max(std::rand() % max_num_points_, 4);

      resize_polygon();

      for (int i = 0; i < raw_points_.size(); ++i)
      {
         raw_points_[i] = edbdmath::random_vec3(-10.f, 10.f);
      }
   }

   char input_name[120];
   sprintf(input_name, "%s num points", prefix_.c_str());
   ImGui::InputInt(input_name, &num_points, 1, 1);
   num_points = std::min(num_points, max_num_points_);
   num_points = std::max(num_points, 3);

   resize_polygon();

   bool vert_changed = false;
   for (int i = 0; i < num_points; ++i)
   {
      char point_label[100];
      sprintf(point_label, "%s %i", prefix_.c_str(), i);
      vert_changed |= ImGui::DragFloat2(point_label, &(raw_points_[i][0]), 0.01f, -10.f, 10.f);
   }

   ImGui::Separator();

   char center_label[100];
   sprintf(center_label, "%s center", prefix_.c_str());
   bool center_changed = ImGui::DragFloat2(
      center_label, &center[0], 0.01f, -10.f, 10.f
   );

   char yaw_label[100];
   sprintf(yaw_label, "%s yaw", prefix_.c_str());
   bool yaw_changed = ImGui::DragFloat(
      yaw_label, &yaw, 0.01f, -2 * M_PI, 2 * M_PI
   );

   bool ui_modified = (
      vert_changed ||
      center_changed ||
      yaw_changed ||
      raw_points_.size() != points.size()
   );

   if (ui_modified)
   {
      update_transformed_points();
   }

   ImGui::Separator();

   return ui_modified;
}

void PolygonUI::resize_polygon(void)
{
   if (raw_points_.size() != num_points)
   {
      raw_points_.resize(num_points);
   }
}

void PolygonUI::update_transformed_points(void)
{
   points.resize(raw_points_.size());

   for (int i = 0; i < points.size(); ++i)
   {
      Matrix33 rot_mat = frd2NedMatrix(0.f, 0.f, yaw);
      points[i] = rot_mat * raw_points_[i] + center;
   }
}
