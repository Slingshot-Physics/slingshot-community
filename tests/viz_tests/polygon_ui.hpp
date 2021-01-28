#ifndef POLYGON_UI_HEADER
#define POLYGON_UI_HEADER

#include "gui_callback_base.hpp"
#include "vector3.hpp"

#include <algorithm>
#include <vector>

class PolygonUI : public viz::GuiCallbackBase
{
   public:

      PolygonUI(const char * prefix, int max_num_points)
         : num_points(3)
         , yaw(0.f)
         , max_num_points_(max_num_points)
         , prefix_(prefix)
      {
         raw_points_.push_back(Vector3(-0.5, 0.f, 0.f));
         raw_points_.push_back(Vector3(0.f, 1.f, 0.f));
         raw_points_.push_back(Vector3(0.5, 0.f, 0.f));

         update_transformed_points();
      }

      void operator()(void);

      bool no_window(void);

      int num_points;

      Vector3 center;

      std::vector<Vector3> points;

      float yaw;

   private:

      int max_num_points_;

      std::string prefix_;

      std::vector<Vector3> raw_points_;

      void resize_polygon(void);

      // Transforms the list of raw points according to the UI's transform.
      void update_transformed_points(void);

};

#endif
