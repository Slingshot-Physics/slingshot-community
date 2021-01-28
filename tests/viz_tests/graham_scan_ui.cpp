#include "graham_scan_ui.hpp"

#include "data_model.h"
#include "default_camera_controller.hpp"
#include "geometry_type_converters.hpp"
#include "logger_utils.hpp"
#include "viz_renderer.hpp"

void MeddlingGui::operator()(void)
{
   ImGui::Begin("Graham scan convex hull");

   no_window();

   ImGui::End();
}

bool MeddlingGui::no_window(void)
{
   bool ui_modified = false;

   ImGui::Text("Clipping polygon");

   ui_modified |= polygon.no_window();

   if (ui_modified)
   {
      for (unsigned int i = 0; i < polygon.points.size(); ++i)
      {
         polygon.points[i][2] = 0.f;
      }
      make_hull();
   }

   return ui_modified;
}

void MeddlingGui::make_hull(void)
{
   geometry::types::polygon50_t temp_poly;
   convert_vec_of_vecs_to_polygon(polygon.points, temp_poly);
   geometry::polygon::convexHull(temp_poly, hull);
   for (int i = 0; i < hull.numVerts; ++i)
   {
      hull.verts[i][2] = 0.f;
   }
}

int main(void)
{
   MeddlingGui gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   int polygon_id = renderer.addSegment(gui.polygon.points, red);

   int hull_id = renderer.addSegment(gui.hull.numVerts, gui.hull.verts, green);

   viz::Camera camera;
   // renderer.initializeCamera(camera);
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   while(true)
   {
      renderer.updateSegment(polygon_id, gui.polygon.points);
      renderer.updateSegment(hull_id, gui.hull.numVerts, gui.hull.verts);

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   std::cout << "Closed the window" << std::endl;

   return 0;
}
