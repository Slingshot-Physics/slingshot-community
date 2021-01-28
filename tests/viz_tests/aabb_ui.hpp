#ifndef AABB_UI_HEADER
#define AABB_UI_HEADER

#include "data_model.h"
#include "geometry_types.hpp"
#include "gui_callback_base.hpp"
#include "vector3.hpp"

class AABBUI : public viz::GuiCallbackBase
{
   public:
      AABBUI(const char * prefix)
         : corners_{
            {-1.f, -1.f, -1.f},
            {1.f, 1.f, 1.f},
         }
         , prefix_(prefix)
      {
         viz_mesh.numTriangles = 12;
         viz_mesh.numVerts = 8;
      }

      void operator()(void);

      bool no_window(void);

      geometry::types::aabb_t aabb;

      data_triangleMesh_t viz_mesh;

   private:
      Vector3 corners_[2];

      std::string prefix_;

      Vector3 center_;

      void update_viz_mesh(void);

};

#endif
