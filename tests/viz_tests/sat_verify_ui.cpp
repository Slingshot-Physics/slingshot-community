#include "sat_verify_ui.hpp"

#include "default_camera_controller.hpp"
#include "viz_renderer.hpp"

bool SatVerify::no_window(void)
{
   ui_modified_ = false;
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
      if (sat_out_.collision)
      {
         std::cout << "collision normal: " << sat_out_.contactNormal << "\n";
      }
   }

   return true;
}

int main(void)
{
   SatVerify gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.setCameraSpeed(0.03f);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   renderer.disableGrid();

   viz::types::vec4_t blue = {0.f, 0.f, 1.f, 1.f};
   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};
   viz::types::vec4_t yellow = {1.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t orange = {1.f, 165.f/255.f, 0.f, 1.f};
   viz::types::vec4_t purple = {160.f/255.f, 32.f/255.f, 240.f/255.f, 1.f};

   int shape_mesh_ids[2] = {
      renderer.addMesh(gui.shape_render_data[0], red, 1),
      renderer.addMesh(gui.shape_render_data[1], green, 1)
   };

   std::vector<int> cylinder_mesh_ids;
   for (unsigned int i = 0; i < gui.cylinder_transforms.size(); ++i)
   {
      cylinder_mesh_ids.push_back(
         renderer.addMesh(
            geometry::mesh::loadDefaultShapeMeshData(
               geometry::types::enumShape_t::CYLINDER, 1.f
            ),
            0
         )
      );
   }

   while(true)
   {
      if (gui.ui_modified())
      {
         for (int i = 0; i < 2; ++i)
         {
            renderer.updateMesh(shape_mesh_ids[i], gui.shape_render_data[i]);
            geometry::types::transform_t trans_M_to_W = gui.shape_loaders[i].trans_B_to_W();
            renderer.updateMeshTransform(shape_mesh_ids[i], trans_M_to_W.translate, trans_M_to_W.rotate, trans_M_to_W.scale);
         }
      }

      if (gui.ui_modified())
      {
         for (unsigned int i = 0; i < cylinder_mesh_ids.size(); ++i)
         {
            renderer.updateMeshTransform(
               cylinder_mesh_ids[i],
               gui.cylinder_transforms[i].translate,
               gui.cylinder_transforms[i].rotate,
               gui.cylinder_transforms[i].scale
            );
            renderer.updateMeshColor(
               cylinder_mesh_ids[i],
               gui.cylinder_colors[i]
            );
         }
      }

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
