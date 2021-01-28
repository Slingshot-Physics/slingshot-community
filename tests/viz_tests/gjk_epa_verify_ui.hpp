#include "gui_callback_base.hpp"
#include "polyhedron_loader_ui.hpp"
#include "vector3.hpp"

#include "md_hull_intersection.hpp"

#include <algorithm>

// Want to generate the convex hull of the MD polyhedron given two polyhedra
// and some transformations.
// Want to verify that the MD hull encapsulates the origin when the bodies
// overlap each other.
class MdHullGui : public viz::GuiCallbackBase
{
   public:
      MdHullGui(void)
         : poly_loaders{
            {"polygon A", Vector3(0.f, 0.f, 2.f)},
            {"polygon B", Vector3(0.f, 0.f, -2.f)}
         }
         , intersection(false)
         , ui_modified_(false)
         , file_path_("")
         , button_color_(0.f, 1.f, 0.f, 1.f)
      {
         for (int i = 0; i < 2; ++i)
         {
            update_render_mesh(i);
         }

         calculateMdHull();
      }

      bool ui_modified(void) const
      {
         return ui_modified_;
      }

      void operator()(void);

      bool no_window(void);

      PolyhedronLoader poly_loaders[2];

      data_triangleMesh_t md_hull_data;

      bool intersection;

   private:

      bool ui_modified_;

      char file_path_[256];

      ImVec4 button_color_;

      void calculateMdHull(void);

      // Load a config file.
      bool load_file(void);

      void update_render_mesh(int index)
      {
         geometry::types::triangleMesh_t temp_mesh = poly_loaders[index].mesh();
         geometry::converters::to_pod(temp_mesh, &render_meshes_data[index]);
      }

   public:

      data_triangleMesh_t render_meshes_data[2];

};
