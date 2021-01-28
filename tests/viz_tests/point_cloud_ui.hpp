#ifndef POINT_CLOUD_HEADER
#define POINT_CLOUD_HEADER

#include "gui_callback_base.hpp"
#include "vector3.hpp"

#include <algorithm>
#include <vector>

class PointCloud : public viz::GuiCallbackBase
{
   public:

      PointCloud(const char * prefix, int max_num_points)
         : num_points(4)
         , min_num_points_(4)
         , max_num_points_(max_num_points)
         , prefix_(prefix)
         , file_path_("")
         , button_color_(0.f, 1.f, 0.f, 1.f)
         , update_fields_(false)
      {
         points.push_back(Vector3(0.f, 1.f, 0.f));
         points.push_back(Vector3(-0.5, 0.f, 0.f));
         points.push_back(Vector3(0.5, 0.f, 0.f));
         points.push_back(Vector3(0.f, 0.f, 0.5f));
      }

      void operator()(void);

      bool no_window(void);

      int num_points;

      std::vector<Vector3> points;

   private:

      int min_num_points_;

      int max_num_points_;

      std::string prefix_;

      char file_path_[256];

      ImVec4 button_color_;

      bool update_fields_;

      void resize_point_cloud(void);

};

#endif
