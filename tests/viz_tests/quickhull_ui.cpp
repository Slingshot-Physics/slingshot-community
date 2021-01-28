#include "quickhull_ui.hpp"

#include "default_camera_controller.hpp"
#include "geometry.hpp"
#include "geometry_type_converters.hpp"
#include "viz_renderer.hpp"

void MeddlingGui::operator()(void)
{
   ImGui::Begin("Quick hull demo");
   no_window();
   ImGui::End();
}

bool MeddlingGui::no_window(void)
{
   rehulled = false;

   if (ImGui::Button("Generate hull!"))
   {
      calculate_hull();
   }

   point_cloud.no_window();

   return rehulled;
}

void MeddlingGui::calculate_hull(void)
{
   // Set up the vertices to call quickhull
   for (int i = 0; i < point_cloud.num_points; ++i)
   {
      md_verts_[i].bodyAVertId = -1;
      md_verts_[i].bodyBVertId = -1;
      md_verts_[i].vert = point_cloud.points[i];
   }

   geometry::mesh::generateHull(point_cloud.num_points, md_verts_, convex_hull_);
   geometry::converters::to_pod(convex_hull_, &convex_hull_data);
   rehulled = true;

   std::cout << "num verts in new hull? " << convex_hull_.numVerts << "\n";

   std::cout << "hull verts:\n";
   for (int i = 0; i < convex_hull_.numVerts; ++i)
   {
      std::cout << "\t" << convex_hull_.verts[i] << "\n";
   }

   std::cout << "num triangles in new hull? " << convex_hull_.numTriangles << "\n";
   std::cout << "hull triangles:\n";
   for (int i = 0; i < convex_hull_.numTriangles; ++i)
   {
      std::cout << "\t" << convex_hull_.triangles[i].vertIds[0] << ", " << convex_hull_.triangles[i].vertIds[1] << ", " << convex_hull_.triangles[i].vertIds[2] << "\n";
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

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.setCameraSpeed(0.01f);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   int hull_id = renderer.addMesh(gui.convex_hull_data, green, 0);

   int cloud_ids[MAX_NUM_POINTS];

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::CUBE, 0.05f
      );

   for (int i = 0; i < MAX_NUM_POINTS; ++i)
   {
      cloud_ids[i] = renderer.addMesh(temp_mesh_data, red, 0);
      renderer.updateMeshTransform(
         cloud_ids[i], Vector3(1000000.f, 0.f, 0.f), identityMatrix(), identityMatrix()
      );
   }

   while (true)
   {
      if (gui.rehulled)
      {
         renderer.updateMesh(hull_id, gui.convex_hull_data);
      }

      for (int i = 0; i < MAX_NUM_POINTS; ++i)
      {
         if (i >= gui.point_cloud.num_points)
         {
            renderer.updateMeshTransform(
               cloud_ids[i], Vector3(1000000.f, 0.f, 0.f), identityMatrix(), identityMatrix()
            );
         }
         else
         {
            renderer.updateMeshTransform(
               cloud_ids[i], gui.point_cloud.points[i], identityMatrix(), identityMatrix()
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
