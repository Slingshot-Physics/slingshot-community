#include "point_cloud_ui.hpp"

#include "data_vector3.h"
#include "geometry_type_converters.hpp"
#include "random_utils.hpp"

#include <random>

void PointCloud::operator()(void)
{
   ImGui::Begin("Point Cloud UI");

   this->no_window();

   ImGui::End();
}

bool PointCloud::no_window(void)
{
   bool ui_modified = false;

   char randomize_name[120];
   sprintf(randomize_name, "%s randomize!", prefix_.c_str());
   if (ImGui::Button(randomize_name))
   {
      ui_modified |= true;
      num_points = std::max(std::rand() % max_num_points_, min_num_points_);

      resize_point_cloud();

      for (int i = 0; i < points.size(); ++i)
      {
         points[i] = edbdmath::random_vec3(-10.f, 10.f);
      }
   }

   char input_name[120];
   sprintf(input_name, "%s num points", prefix_.c_str());
   ui_modified |= ImGui::InputInt(input_name, &num_points, 1, 1);
   num_points = std::min(num_points, max_num_points_);
   num_points = std::max(num_points, 4);

   resize_point_cloud();

   for (int i = 0; i < num_points; ++i)
   {
      char point_label[100];
      sprintf(point_label, "%s point %i position", prefix_.c_str(), i);
      ui_modified |= ImGui::DragFloat3(point_label, &(points[i][0]), 0.01f, -10.f, 10.f);
   }

   return ui_modified;
}

void PointCloud::resize_point_cloud(void)
{
   if (points.size() != num_points)
   {
      points.resize(num_points);
   }
}
