#include "polygon_segment_ui.hpp"

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

   ui_modified |= ImGui::DragFloat2("segment pt a", &(subject_segment.points[0][0]), 0.025f);
   ui_modified |= ImGui::DragFloat2("segment pt b", &(subject_segment.points[1][0]), 0.025f);

   subject_segment.points[0][2] = 0.f;
   subject_segment.points[1][2] = 0.f;

   if (ui_modified)
   {
      clip_shapes();
   }

   ImGui::Text("Num intersection verts: %lu", intersection.size());

   return ui_modified;
}

void MeddlingGui::clip_shapes(void)
{
   geometry::types::polygon50_t clip_poly_type;
   convert_vec_of_vecs_to_polygon(clip_poly.points, clip_poly_type);

   geometry::types::segment_t output_segment;
   bool clipped = geometry::polygon::clipSegment(
      clip_poly_type, subject_segment, output_segment
   );

   if (clipped)
   {
      intersection.resize(2);
      for (int i = 0; i < 2; ++i)
      {
         intersection[i] = output_segment.points[i];
         intersection[i][2] = 4.f;
      }
   }
   else
   {
      intersection.resize(0);
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

   int subject_id = renderer.addSegment(2, gui.subject_segment.points, red);

   int clipped_poly_id = renderer.addSegment(gui.intersection, cyan);

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   while(true)
   {
      renderer.updateSegment(clip_id, gui.clip_poly.points);
      renderer.updateSegment(subject_id, 2, gui.subject_segment.points);
      renderer.updateSegment(clipped_poly_id, gui.intersection);

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   std::cout << "Closed the window" << std::endl;
   return 0;
}
