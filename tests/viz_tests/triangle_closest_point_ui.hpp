#ifndef TRIANGLE_CLOSEST_POINT_UI_HEADER
#define TRIANGLE_CLOSEST_POINT_UI_HEADER

#include "gui_callback_base.hpp"
#include "triangle_ui.hpp"
#include "vector3.hpp"

class TriangleClosestPointUI : public viz::GuiCallbackBase
{
   public:
      TriangleClosestPointUI(void)
         : triangle_ui("alg")
      { }

      void operator()(void);

      bool no_window(void);

      TriangleUI triangle_ui;

      Vector3 query_point;

      geometry::types::pointBaryCoord_t closest_bary_pt;

      Vector4 query_bary_pt;
};

#endif
