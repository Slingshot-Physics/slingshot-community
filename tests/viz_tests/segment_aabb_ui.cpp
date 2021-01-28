#include "segment_aabb_ui.hpp"

#include "default_camera_controller.hpp"
#include "mesh.hpp"
#include "segment.hpp"
#include "viz_renderer.hpp"

void MeddlingGui::operator()(void)
{
   ImGui::Begin("Closest points between segment and AABB");
   no_window();
   ImGui::End();
}

bool MeddlingGui::no_window(void)
{
   ui_modified_ = false;

   segment_loader.line_type = 2;

   ui_modified_ |= segment_loader.no_window();

   ui_modified_ |= aabb_loader.no_window();

   segment_loader.line_type = 2;

   if (ui_modified_)
   {
      calculate_closest_points();
   }

   return ui_modified_;
}

void MeddlingGui::calculate_closest_points(void)
{
   closest_points = geometry::segment::closestPointsToAabb(
      segment_loader.line_points[0],
      segment_loader.line_points[1],
      aabb_loader.aabb
   );
}

int main(void)
{
   MeddlingGui gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.setCameraSpeed(0.02f);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t blue = {0.f, 0.f, 1.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   int segment_id = renderer.addSegment(
      2, gui.segment_loader.line_points, green
   );

   int aabb_id = renderer.addMesh(gui.aabb_loader.viz_mesh, blue, 0);

   data_triangleMesh_t temp_sphere_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );

   int closest_aabb_point_ids[2] = {
      renderer.addMesh(temp_sphere_mesh_data, magenta, 1),
      renderer.addMesh(temp_sphere_mesh_data, magenta, 1)
   };

   int closest_segment_point_ids[2] = {
      renderer.addMesh(temp_sphere_mesh_data, cyan, 1),
      renderer.addMesh(temp_sphere_mesh_data, cyan, 1)
   };

   bool first_run = true;

   while (true)
   {
      if (gui.ui_modified() || first_run)
      {
         first_run = false;
         renderer.updateMesh(aabb_id, gui.aabb_loader.viz_mesh);
         renderer.updateSegment(segment_id, 2, gui.segment_loader.line_points);
         for (int i = 0; i < 2; ++i)
         {
            renderer.updateMeshTransform(
               closest_aabb_point_ids[i],
               gui.closest_points.otherPoints[i],
               identityMatrix(),
               identityMatrix()
            );

            renderer.updateMeshTransform(
               closest_segment_point_ids[i],
               gui.closest_points.segmentPoints[i],
               identityMatrix(),
               identityMatrix()
            );
         }

         if (gui.closest_points.numPairs == 1)
         {
            renderer.disableMesh(closest_aabb_point_ids[1]);
            renderer.disableMesh(closest_segment_point_ids[1]);
         }
         else
         {
            renderer.enableMesh(closest_aabb_point_ids[1]);
            renderer.enableMesh(closest_segment_point_ids[1]);
         }
      }

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
