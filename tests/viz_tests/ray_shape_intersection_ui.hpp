#ifndef RAY_SHAPE_INTERSECTION_UI_HEADER
#define RAY_SHAPE_INTERSECTION_UI_HEADER

#include "geometry_type_converters.hpp"
#include "geometry_types.hpp"

#include "line_loader_ui.hpp"
#include "shape_loader_ui.hpp"

class RaycastShapeUI: public viz::GuiCallbackBase
{
   public:
      RaycastShapeUI(void)
         : ui_modified_(false)
         , shape_loader("shape", {0.f, 0.f, 0.f})
         , ray("ray")
      {
         update_render_mesh();
      }

      void operator()(void);

      bool no_window(void);

      bool ui_modified(void)
      {
         return ui_modified_;
      }

      const data_triangleMesh_t & mesh(void) const
      {
         return render_mesh_data_;
      }

      ShapeLoader shape_loader;

      LineLoader ray;

      geometry::types::raycastResult_t raycast;

   private:
      bool ui_modified_;

      data_triangleMesh_t render_mesh_data_;

      void update_render_mesh(void)
      {
         geometry::types::triangleMesh_t temp_mesh = shape_loader.mesh();
         geometry::converters::to_pod(temp_mesh, &render_mesh_data_);
      }

};

#endif
