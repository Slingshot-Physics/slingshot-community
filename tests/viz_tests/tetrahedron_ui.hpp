#ifndef TETRAHEDRON_UI_HEADER
#define TETRAHEDRON_UI_HEADER

#include "data_model.h"
#include "geometry_types.hpp"
#include "gui_callback_base.hpp"
#include "vector3.hpp"

class TetrahedronUI : public viz::GuiCallbackBase
{
   public:

      TetrahedronUI(const char * prefix)
         : num_points_(4)
         , prefix_(prefix)
      {
         raw_tetrahedron_.verts[0] = Vector3(-0.5, 0.f, 0.f);
         raw_tetrahedron_.verts[1] = Vector3(0.f, 1.f, 0.f);
         raw_tetrahedron_.verts[2] = Vector3(0.5, 0.f, 0.f);
         raw_tetrahedron_.verts[3] = Vector3(0.0, 0.f, 1.f);
         update_transformed_points();
      }

      void operator()(void);

      bool no_window(void);

      Vector3 center;

      Vector3 rpy;

      geometry::types::tetrahedron_t tetrahedron;

      data_triangleMesh_t viz_mesh;

   private:
      unsigned int num_points_;

      std::string prefix_;

      geometry::types::tetrahedron_t raw_tetrahedron_;

      // Transforms the list of raw points according to the UI's transform.
      void update_transformed_points(void);
};

#endif
