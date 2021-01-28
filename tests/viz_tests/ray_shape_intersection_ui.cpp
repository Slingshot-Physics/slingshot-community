#include "ray_shape_intersection_ui.hpp"

#include "default_camera_controller.hpp"
#include "raycast.hpp"
#include "viz_renderer.hpp"

void RaycastShapeUI::operator()(void)
{
   ImGui::Begin("raycast/shape UI");
   no_window();
   ImGui::End();
}

bool RaycastShapeUI::no_window(void)
{
   ui_modified_ = false;
   ui_modified_ |= shape_loader.no_window();
   ui_modified_ |= ray.no_window();

   if (ui_modified_)
   {
      update_render_mesh();

      switch (shape_loader.shape().shapeType)
      {
         case geometry::types::enumShape_t::SPHERE:
         {
            raycast = geometry::raycast(
               ray.line_points[0],
               ray.line_points[1],
               shape_loader.trans_B_to_W(),
               shape_loader.shape().sphere
            );
            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            raycast = geometry::raycast(
               ray.line_points[0],
               ray.line_points[1],
               shape_loader.trans_B_to_W(),
               shape_loader.shape().cylinder
            );
            break;
         }
         case geometry::types::enumShape_t::CAPSULE:
         {
            raycast = geometry::raycast(
               ray.line_points[0],
               ray.line_points[1],
               shape_loader.trans_B_to_W(),
               shape_loader.shape().capsule
            );
            break;
         }
         case geometry::types::enumShape_t::CUBE:
         {
            raycast = geometry::raycast(
               ray.line_points[0],
               ray.line_points[1],
               shape_loader.trans_B_to_W(),
               shape_loader.shape().cube
            );
            break;
         }
         default:
         {
            raycast.hit = false;
            std::cout << "Don't know how to do an intersection with shape type " << static_cast<int>(shape_loader.shape().shapeType) << "\n";
            break;
         }
      }
   }

   return ui_modified_;
}

int main(void)
{
   RaycastShapeUI gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.cameraSpeed() = 0.01f;
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   int shape_id = renderer.addMesh(gui.mesh(), green, 0);

   int ray_id = renderer.addSegment(2, gui.ray.line_points, red);

   data_triangleMesh_t sphere_mesh_data = geometry::mesh::loadDefaultShapeMeshData(geometry::types::enumShape_t::SPHERE, 0.05f);
   int near_hit_id = renderer.addMesh(sphere_mesh_data, magenta, 1);
   int far_hit_id = renderer.addMesh(sphere_mesh_data, magenta, 1);

   Vector3 segment_verts[2] = {
      {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}
   };

   while (true)
   {
      if (gui.ui_modified())
      {
         renderer.updateMesh(shape_id, gui.mesh());
         segment_verts[0] = gui.ray.start;
         segment_verts[1] = gui.ray.start + 100.f * gui.ray.slope;
         renderer.updateSegment(ray_id, 2, segment_verts);
      }

      if (gui.raycast.hit)
      {
         renderer.updateMeshColor(shape_id, cyan);
      }
      else
      {
         renderer.updateMeshColor(shape_id, green);
      }

      if (gui.raycast.numHits == 0)
      {
         renderer.disableMesh(near_hit_id);
         renderer.disableMesh(far_hit_id);
      }
      else if (gui.raycast.numHits == 1)
      {
         renderer.enableMesh(near_hit_id);
         renderer.disableMesh(far_hit_id);
      }
      else
      {
         renderer.enableMesh(near_hit_id);
         renderer.enableMesh(far_hit_id);
      }

      renderer.updateMeshTransform(near_hit_id, gui.raycast.hits[0], identityMatrix(), identityMatrix());
      renderer.updateMeshTransform(far_hit_id, gui.raycast.hits[1], identityMatrix(), identityMatrix());

      geometry::types::transform_t trans_B_to_W = gui.shape_loader.trans_B_to_W();
      renderer.updateMeshTransform(
         shape_id, trans_B_to_W.translate, trans_B_to_W.rotate, trans_B_to_W.scale
      );
      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
