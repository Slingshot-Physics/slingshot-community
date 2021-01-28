#include <cmath>
#include <iostream>
#include <thread>

#include "data_model.h"
#include "default_camera_controller.hpp"
#include "geometry.hpp"
#include "viz_renderer.hpp"

#include "segment_segment_ui.hpp"

void MeddlingGui::operator()(void)
{
   ImGui::Begin("Line and segment manipulatrix");
   no_window();
   ImGui::End();
}

bool MeddlingGui::no_window(void)
{
   ui_modified_ = false;
   segment_loader_a.line_type = 2;

   segment_loader_b.line_type = 2;

   ui_modified_ |= segment_loader_a.no_window();

   ui_modified_ |= segment_loader_b.no_window();

   segment_loader_a.line_type = 2;

   segment_loader_b.line_type = 2;

   if (ui_modified_)
   {
      calculate_closest_points();
   }

   return ui_modified_;
}

void MeddlingGui::calculate_closest_points(void)
{
   const auto closest_segment_points = geometry::segment::closestPointsToSegment(
      segment_loader_a.line_points[0],
      segment_loader_a.line_points[1],
      segment_loader_b.line_points[0],
      segment_loader_b.line_points[1],
      1e-7f
   );

   closest_point_pairs.num_points = closest_segment_points.numPairs;
   for (int i = 0; i < 2; ++i)
   {
      closest_point_pairs.seg_a_points[i] = closest_segment_points.segmentPoints[i];
      closest_point_pairs.seg_b_points[i] = closest_segment_points.otherPoints[i];
   }
}

int main(int argc, char * argv[])
{
   MeddlingGui gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   int segment_a_id = renderer.addSegment(
      2, gui.segment_loader_a.line_points, red
   );

   int segment_b_id = renderer.addSegment(
      2, gui.segment_loader_b.line_points, green
   );

   data_triangleMesh_t temp_sphere_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );

   int seg_a_closest_point_ids[2] = {
      renderer.addMesh(
         temp_sphere_mesh_data, cyan, 0
      ),
      renderer.addMesh(
         temp_sphere_mesh_data, cyan, 0
      )
   };

   int seg_b_closest_point_ids[2] = {
      renderer.addMesh(
         temp_sphere_mesh_data, magenta, 0
      ),
      renderer.addMesh(
         temp_sphere_mesh_data, magenta, 0
      )
   };

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.setCameraSpeed(0.02f);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   while(true)
   {
      if (gui.ui_modified())
      {
         renderer.updateSegment(
            segment_a_id, 2, gui.segment_loader_a.line_points
         );

         renderer.updateSegment(
            segment_b_id, 2, gui.segment_loader_b.line_points
         );

         for (int i = 0; i < 2; ++i)
         {
            renderer.updateMeshTransform(
               seg_a_closest_point_ids[i],
               gui.closest_point_pairs.seg_a_points[i],
               identityMatrix(),
               identityMatrix()
            );
            renderer.updateMeshTransform(
               seg_b_closest_point_ids[i],
               gui.closest_point_pairs.seg_b_points[i],
               identityMatrix(),
               identityMatrix()
            );
         }

         if (gui.closest_point_pairs.num_points == 1)
         {
            renderer.disableMesh(seg_a_closest_point_ids[1]);
            renderer.disableMesh(seg_b_closest_point_ids[1]);
         }
         else
         {
            renderer.enableMesh(seg_a_closest_point_ids[1]);
            renderer.enableMesh(seg_b_closest_point_ids[1]);
         }
      }

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   std::cout << "Closed the window" << std::endl;
   return 0;
}
