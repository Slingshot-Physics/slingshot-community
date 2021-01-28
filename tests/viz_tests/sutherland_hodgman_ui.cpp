#include "sutherland_hodgman_ui.hpp"

#include "data_model.h"
#include "default_camera_controller.hpp"
#include "geometry_type_converters.hpp"
#include "logger_utils.hpp"
#include "viz_renderer.hpp"

void MeddlingGui::operator()(void)
{
   ImGui::Begin("Polygon clipper/intersectrix");
   no_window();
   ImGui::End();
}

bool MeddlingGui::no_window(void)
{
   bool ui_modified = false;
   ImGui::Text("Clipping polygon");

   ui_modified |= clip_poly.no_window();

   ImGui::Text("Subject polygon");

   ui_modified |= subject_poly.no_window();

   if (ui_modified)
   {
      clip_polygons();
   }

   ImGui::Text("Num intersection verts: %lu", intersection.size());

   return ui_modified;
}

void MeddlingGui::clip_polygons(void)
{
   geometry::types::polygon50_t clip_poly_type;
   convert_vec_of_vecs_to_polygon(clip_poly.points, clip_poly_type);
   geometry::types::polygon50_t subject_poly_type;
   convert_vec_of_vecs_to_polygon(subject_poly.points, subject_poly_type);

   geometry::types::polygon50_t output_poly;
   geometry::polygon::clipSutherlandHodgman(
      clip_poly_type, subject_poly_type, output_poly
   );

   if (output_poly.numVerts != intersection.size())
   {
      intersection.resize(output_poly.numVerts);
   }

   for (int i = 0; i < output_poly.numVerts; ++i)
   {
      intersection[i] = output_poly.verts[i];
      intersection[i][2] = 4.f;
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

   int clip_id = renderer.addSegment(gui.clip_poly.points, green);

   int subject_id = renderer.addSegment(gui.subject_poly.points, red);

   int clipped_poly_id = renderer.addSegment(gui.intersection, cyan);

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   while(true)
   {
      renderer.updateSegment(clip_id, gui.clip_poly.points);
      renderer.updateSegment(subject_id, gui.subject_poly.points);
      renderer.updateSegment(clipped_poly_id, gui.intersection);

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   std::cout << "Closed the window" << std::endl;
   return 0;
}
