#include "gui_callback_base.hpp"

#include "slingshot_types.hpp"
#include "point_cloud_ui.hpp"

#include <iostream>

#define MAX_NUM_POINTS 500

class MeddlingGui : public viz::GuiCallbackBase
{
   public:
      MeddlingGui(void)
         : rehulled(false)
         , point_cloud("cloud", MAX_NUM_POINTS)
      {
         calculate_hull();
         std::cout << "num verts in the convex hull? " << convex_hull_.numVerts << "\n";
      }

      void operator()(void);

      bool no_window(void);

      bool rehulled;

      PointCloud point_cloud;

      data_triangleMesh_t convex_hull_data;

   private:
      void calculate_hull(void);

      geometry::types::minkowskiDiffVertex_t md_verts_[MAX_NUM_POINTS];

      geometry::types::triangleMesh_t convex_hull_;

};
