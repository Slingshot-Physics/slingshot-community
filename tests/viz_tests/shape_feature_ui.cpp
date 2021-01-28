#include "shape_feature_ui.hpp"

#include "shape_features.hpp"

#include "default_camera_controller.hpp"
#include "gauss_map.hpp"
#include "viz_renderer.hpp"

#include <chrono>
#include <thread>
#include <vector>

bool MeshFeature::no_window(void)
{
   ui_modified_ = false;
   ui_modified_ |= shape_loader.no_window();

   ui_modified_ |= transform_ui.no_window();

   query_plane_.point = transform_ui.trans_B_to_W().translate;
   query_plane_.normal = transform_ui.trans_B_to_W().rotate * Vector3{0.f, 0.f, 1.f};

   if (ui_modified_)
   {
      const geometry::types::shape_t shape = shape_loader.shape();
      switch(shape.shapeType)
      {
         case geometry::types::enumShape_t::CAPSULE:
         {
            best_feature = geometry::mostParallelFeature(
               shape_loader.trans_B_to_W(),
               shape.capsule,
               query_plane_
            ).shape;
            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            best_feature = geometry::mostParallelFeature(
               shape_loader.trans_B_to_W(),
               shape.cylinder,
               query_plane_
            ).shape;
            break;
         }
         case geometry::types::enumShape_t::CUBE:
         {
            best_feature = geometry::mostParallelFeature(
               shape_loader.trans_B_to_W(),
               shape.cube,
               query_plane_
            ).shape;
            break;
         }
         case geometry::types::enumShape_t::SPHERE:
         {

         }
         default:
         {
            std::cout << "no feature function for shape type: " << static_cast<int>(shape_loader.shape().shapeType) << "\n";
            best_feature.numVerts = 0;
         }
      }

      update_render_meshes();
   }

   return ui_modified_;
}

int main(void)
{
   MeshFeature gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.setCameraSpeed(0.03f);
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

   int poly_mesh_id = renderer.addMesh(gui.shape_render_data, green, 1);

   int plane_mesh_id = renderer.addMesh(gui.plane_render_data, blue, 0);

   std::vector<Vector3> segment_points = {
      {0.f, 0.f, 0.f},
      {0.f, 0.f, 0.f}
   };

   int mesh_normal_segment_id = renderer.addSegment(segment_points, yellow);

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.05f
      );

   std::vector<int> face_point_ids;
   for (int i = 0; i < 50; ++i)
   {
      face_point_ids.push_back(
         renderer.addMesh(temp_mesh_data, magenta, 1)
      );
   }

   while (true)
   {
      if (gui.ui_modified())
      {
         renderer.updateMesh(poly_mesh_id, gui.shape_render_data);

         segment_points[0] = gui.query_plane().point;
         segment_points[1] = gui.query_plane().point + gui.query_plane().normal;
         renderer.updateSegment(mesh_normal_segment_id, segment_points);
      }

      renderer.updateMeshTransform(
         poly_mesh_id,
         gui.shape_loader.trans_B_to_W().translate,
         gui.shape_loader.trans_B_to_W().rotate,
         gui.shape_loader.trans_B_to_W().scale
      );

      renderer.updateMeshTransform(
         plane_mesh_id,
         gui.transform_ui.trans_B_to_W().translate,
         gui.transform_ui.trans_B_to_W().rotate,
         gui.transform_ui.trans_B_to_W().scale
      );

      for (int i = 0; i < 50; ++i)
      {
         if (i >= gui.best_feature.numVerts)
         {
            renderer.disableMesh(face_point_ids[i]);
         }
         else
         {
            renderer.updateMeshTransform(face_point_ids[i], gui.best_feature.verts[i], identityMatrix(), identityMatrix());
            renderer.enableMesh(face_point_ids[i]);
         }
      }

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(2));
   }

   return 0;
}
