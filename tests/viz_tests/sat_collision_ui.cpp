#include "sat_collision_ui.hpp"

#include "default_camera_controller.hpp"
#include "viz_renderer.hpp"

bool SatCollision::no_window(void)
{
   ui_modified_ = false;
   ui_modified_ = ImGui::Checkbox("Show SAT manifold", &sat_manifold_);
   ui_modified_ |= shape_loaders[0].no_window();
   ui_modified_ |= shape_loaders[1].no_window();

   if (ui_modified_)
   {
      update_render_meshes();
   }

   if (ui_modified_)
   {
      const geometry::types::shape_t shapes[2] = {
         shape_loaders[0].shape(), shape_loaders[1].shape()
      };

      collision_multiplexor();
      update_aabb_mesh();
      if (sat_out_.collision)
      {
         std::cout << "collision normal: " << sat_out_.contactNormal << "\n";
      }
   }

   return ui_modified_;
}

int main(void)
{
   SatCollision gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.setCameraSpeed(0.015f);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   viz::types::vec4_t blue = {0.f, 0.f, 1.f, 1.f};
   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};
   viz::types::vec4_t yellow = {1.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t orange = {1.f, 165.f/255.f, 0.f, 1.f};
   viz::types::vec4_t purple = {160.f/255.f, 32.f/255.f, 240.f/255.f, 1.f};

   int shape_a_aabb_id = renderer.addMesh(
      gui.shape_a_aabb_mesh_data, orange, 1
   );

   int shape_mesh_ids[2] = {
      renderer.addMesh(gui.shape_render_data[0], red, 1),
      renderer.addMesh(gui.shape_render_data[1], green, 1)
   };

   data_triangleMesh_t contact_sphere = geometry::mesh::loadDefaultShapeMeshData(
      geometry::types::enumShape_t::SPHERE, 0.05f
   );

   int contact_points_a[4] = {
      renderer.addMesh(
         contact_sphere, blue, 1
      ),
      renderer.addMesh(
         contact_sphere, blue, 1
      ),
      renderer.addMesh(
         contact_sphere, blue, 1
      ),
      renderer.addMesh(
         contact_sphere, blue, 1
      )
   };

   int contact_points_b[4] = {
      renderer.addMesh(
         contact_sphere, yellow, 1
      ),
      renderer.addMesh(
         contact_sphere, yellow, 1
      ),
      renderer.addMesh(
         contact_sphere, yellow, 1
      ),
      renderer.addMesh(
         contact_sphere, yellow, 1
      )
   };

   Vector3 collision_normal_segment[2] = {
      {0.f, 0.f, 0.f},
      {0.f, 0.f, 0.f}
   };

   int collision_normal_id = renderer.addSegment(2, collision_normal_segment, yellow);

   while(true)
   {
      for (int i = 0; i < 2; ++i)
      {
         if (gui.ui_modified())
         {
            renderer.updateMesh(shape_mesh_ids[i], gui.shape_render_data[i]);
         }
         geometry::types::transform_t trans_M_to_W = gui.shape_loaders[i].trans_B_to_W();
         renderer.updateMeshTransform(shape_mesh_ids[i], trans_M_to_W.translate, trans_M_to_W.rotate, trans_M_to_W.scale);
      }

      if (gui.ui_modified())
      {
         renderer.updateMesh(shape_a_aabb_id, gui.shape_a_aabb_mesh_data);
         if (gui.sat_out().collision)
         {
            for (int i = 0; i < 4; ++i)
            {
               if (i < gui.contact_manifold.numContacts)
               {
                  renderer.enableMesh(contact_points_a[i]);
                  renderer.enableMesh(contact_points_b[i]);
               }
               else
               {
                  renderer.disableMesh(contact_points_a[i]);
                  renderer.disableMesh(contact_points_b[i]);
               }

               renderer.updateMeshTransform(
                  contact_points_a[i],
                  gui.contact_manifold.bodyAContacts[i],
                  identityMatrix(),
                  identityMatrix()
               );

               renderer.updateMeshTransform(
                  contact_points_b[i],
                  gui.contact_manifold.bodyBContacts[i],
                  identityMatrix(),
                  identityMatrix()
               );
            }

            renderer.enableMesh(collision_normal_id);
            collision_normal_segment[0] = gui.sat_out().deepestPointsA[0];
            collision_normal_segment[1] = gui.sat_out().deepestPointsA[0] + gui.sat_out().contactNormal;
            renderer.updateSegment(collision_normal_id, 2, collision_normal_segment);
         }
         else
         {
            renderer.disableMesh(collision_normal_id);

            for (int i = 0; i < 4; ++i)
            {
               renderer.disableMesh(contact_points_a[i]);
               renderer.disableMesh(contact_points_b[i]);
            }
         }
      }

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
