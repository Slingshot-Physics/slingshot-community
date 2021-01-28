#ifndef RAY_INTERSECTION_UI_HEADER
#define RAY_INTERSECTION_UI_HEADER

#include "aabb_ui.hpp"
#include "geometry_type_converters.hpp"
#include "line_loader_ui.hpp"
#include "polyhedron_loader_ui.hpp"
#include "triangle_ui.hpp"

class MeddlingGui : public viz::GuiCallbackBase
{

   public:
      MeddlingGui(void)
         : ray("ray")
         , triangle("triangle")
         , aabb("aabb")
         , poly_loader("polyhedron", Vector3(0.f, 0.f, 0.f))
         , triangle_touch_point(1e7f, 0.f, 0.f)
         , triangle_touching(false)
         , aabb_touch_point(1e7f, 0.f, 0.f)
         , aabb_touching(false)
         , mesh_touch_point(1e7f, 0.f, 0.f)
         , mesh_touching(false)
         , ui_modified_(false)
      {
         geometry::types::triangleMesh_t trans_polyhedron(poly_loader.mesh());
         geometry::mesh::applyTransformation(poly_loader.trans_B_to_W(), trans_polyhedron);
         geometry::converters::to_pod(trans_polyhedron, &render_mesh_data);
      }

      void operator()(void);

      bool no_window(void);

      bool ui_modified(void)
      {
         return ui_modified_;
      }

      LineLoader ray;

      TriangleUI triangle;

      AABBUI aabb;

      PolyhedronLoader poly_loader;

      data_triangleMesh_t render_mesh_data;

      Vector3 triangle_touch_point;

      bool triangle_touching;

      Vector3 aabb_touch_point;

      bool aabb_touching;

      Vector3 mesh_touch_point;

      bool mesh_touching;

   private:

      bool ui_modified_;

      void calculate_intersections(void);

};

#endif
