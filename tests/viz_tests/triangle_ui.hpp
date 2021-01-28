#ifndef POLYGON_UI_HEADER
#define POLYGON_UI_HEADER

#include "data_model.h"
#include "geometry_types.hpp"
#include "gui_callback_base.hpp"
#include "vector3.hpp"

class TriangleUI : public viz::GuiCallbackBase
{
   public:

      TriangleUI(const char * prefix)
         : num_points_(3)
         , prefix_(prefix)
      {

         raw_triangle_.verts[0] = Vector3(-0.5, 0.f, 0.f);
         raw_triangle_.verts[1] = Vector3(0.f, 1.f, 0.f);
         raw_triangle_.verts[2] = Vector3(0.5, 0.f, 0.f);

         update_transformed_points();
      }

      void operator()(void);

      bool no_window(void);

      Vector3 center;

      Vector3 rpy;

      geometry::types::triangle_t triangle;

      data_triangleMesh_t viz_mesh;

   private:

      const int num_points_;

      std::string prefix_;

      geometry::types::triangle_t raw_triangle_;

      void resize_polygon(void);

      // Load a config file.
      bool load_file(void);

      // Transforms the list of raw points according to the UI's transform.
      void update_transformed_points(void);

};

#endif
