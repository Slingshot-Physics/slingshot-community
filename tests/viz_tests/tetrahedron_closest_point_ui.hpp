#ifndef TETRAHEDRON_CLOSEST_POINT_UI_HEADER
#define TETRAHEDRON_CLOSEST_POINT_UI_HEADER

#include "gui_callback_base.hpp"
#include "tetrahedron_ui.hpp"
#include "vector3.hpp"

class TetrahedronClosestPointUI : public viz::GuiCallbackBase
{
   public:
      TetrahedronClosestPointUI(void)
         : tetrahedron_ui("alg")      
      { }

      void operator()(void);

      bool no_window(void);

      TetrahedronUI tetrahedron_ui;

      Vector3 query_point;

      geometry::types::pointBaryCoord_t closest_bary_pt;

      Vector4 query_bary_pt;
};

#endif
