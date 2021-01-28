#include <cmath>
#include <iostream>
#include <thread>

#include "data_model.h"
#include "default_camera_controller.hpp"
#include "slingshot_type_converters.hpp"
#include "geometry.hpp"
#include "logger_utils.hpp"
#include "viz_renderer.hpp"

#include "segment_line_ui.hpp"

void MeddlingGui::operator()(void)
{
   ImGui::Begin("Line and segment manipulatrix");
   no_window();
   ImGui::End();
}

bool MeddlingGui::no_window(void)
{
   bool ui_modified = false;
   line_loader.line_type = 0;

   segment_loader.line_type = 2;

   ui_modified |= line_loader.no_window();

   ui_modified |= segment_loader.no_window();

   line_loader.line_type = 0;

   segment_loader.line_type = 2;

   if (ui_modified)
   {
      calculate_intersection();
   }

   return ui_modified;
}

void MeddlingGui::calculate_intersection(void)
{
   Vector3 p, q;
   int intersect_type = geometry::line::segmentIntersection(
      line_loader.start,
      line_loader.start + line_loader.slope,
      segment_loader.line_points[0],
      segment_loader.line_points[1],
      p,
      q
   );

   intersection = intersect_type > 0;
   if (intersection)
   {
      touch_point = p;
   }
   else
   {
      touch_point[0] = 10000000.f;
   }
}

int main(int argc, char * argv[])
{
   MeddlingGui gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   unsigned int line_id = renderer.addSegment(
      2, gui.line_loader.line_points, green
   );
   unsigned int segment_id = renderer.addSegment(
      2, gui.segment_loader.line_points, red
   );

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::CUBE, 0.2f
      );

   unsigned int touch_pt_id = renderer.addMesh(
      temp_mesh_data, magenta, 0
   );

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   while(true)
   {
      renderer.updateMeshTransform(touch_pt_id, gui.touch_point, identityMatrix(), identityMatrix());
      renderer.updateSegment(segment_id, 2, gui.segment_loader.line_points);
      renderer.updateSegment(line_id, 2, gui.line_loader.line_points);

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   std::cout << "Closed the window" << std::endl;
   return 0;
}
