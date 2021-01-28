#include "mesh_feature_ui.hpp"

#include "shape_features.hpp"

#include "default_camera_controller.hpp"
#include "gauss_map.hpp"
#include "viz_renderer.hpp"

#include <vector>

bool MeshFeature::no_window(void)
{
   bool ui_modified = false;
   ui_modified |= poly_loader.no_window();

   ui_modified |= transform_ui.no_window();

   query_plane_.point = transform_ui.trans_B_to_W().translate;
   query_plane_.normal = transform_ui.trans_B_to_W().rotate * Vector3{0.f, 0.f, 1.f};

   if (ui_modified)
   {
      gauss_map_ = geometry::mesh::loadGaussMap(poly_loader.shape_type());
      best_feature = geometry::mostParallelFeature(
         poly_loader.trans_B_to_W(), gauss_map_, query_plane_
      ).shape;

      std::cout << "feature size: " << best_feature.numVerts << "\n";

      update_render_meshes();
   }

   return ui_modified;
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

   int poly_mesh_id = renderer.addMesh(gui.polyhedron_render_data, green, 1);

   int plane_mesh_id = renderer.addMesh(gui.plane_render_data, blue, 0);

   std::vector<Vector3> segment_points = {
      {0.f, 0.f, 0.f},
      {0.f, 0.f, 0.f}
   };

   int mesh_normal_segment_id = renderer.addSegment(segment_points, yellow);

   std::vector<int> face_point_ids;

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.05f
      );

   for (int i = 0; i < 50; ++i)
   {
      face_point_ids.push_back(
         renderer.addMesh(temp_mesh_data, magenta, 1)
      );
   }

   while (true)
   {
      if (gui.poly_loader.polyhedron_changed())
      {
         renderer.updateMesh(poly_mesh_id, gui.polyhedron_render_data);
      }

      segment_points[0] = gui.query_plane().point;
      segment_points[1] = gui.query_plane().point + gui.query_plane().normal;
      renderer.updateSegment(mesh_normal_segment_id, segment_points);

      renderer.updateMeshTransform(
         poly_mesh_id,
         gui.poly_loader.trans_B_to_W().translate,
         gui.poly_loader.trans_B_to_W().rotate,
         gui.poly_loader.trans_B_to_W().scale
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
            renderer.enableMesh(face_point_ids[i]);
         }
         renderer.updateMeshTransform(face_point_ids[i], gui.best_feature.verts[i], identityMatrix(), identityMatrix());
      }

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
