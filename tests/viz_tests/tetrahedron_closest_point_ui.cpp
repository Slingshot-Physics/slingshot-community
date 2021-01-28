#include "tetrahedron_closest_point_ui.hpp"

#include "mesh.hpp"
#include "tetrahedron.hpp"

#include "default_camera_controller.hpp"
#include "viz_renderer.hpp"

void TetrahedronClosestPointUI::operator()(void)
{
   ImGui::Begin("tetrahedron closest point UI");
   no_window();
   ImGui::End();
}

bool TetrahedronClosestPointUI::no_window(void)
{
   bool ui_modified = false;
   ui_modified |= tetrahedron_ui.no_window();

   ui_modified |= ImGui::DragFloat3("query point", &(query_point[0]), 0.025f, -32.f, 32.f, "%0.7f");

   ImGui::Text("closest point bary coord:\n[%f, %f, %f, %f]", closest_bary_pt.bary[0], closest_bary_pt.bary[1], closest_bary_pt.bary[2], closest_bary_pt.bary[3]);
   ImGui::Text("query point bary coord:\n[%f, %f, %f, %f]", query_bary_pt[0], query_bary_pt[1], query_bary_pt[2], query_bary_pt[3]);

   if (ui_modified)
   {
      query_bary_pt = geometry::tetrahedron::baryCoords(tetrahedron_ui.tetrahedron, query_point);
      closest_bary_pt = geometry::tetrahedron::closestPointToPoint(
         tetrahedron_ui.tetrahedron.verts[0],
         tetrahedron_ui.tetrahedron.verts[1],
         tetrahedron_ui.tetrahedron.verts[2],
         tetrahedron_ui.tetrahedron.verts[3],
         query_point
      );
   }

   return ui_modified;
}

int main(void)
{
   TetrahedronClosestPointUI gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   int tetrahedron_id = renderer.addMesh(gui.tetrahedron_ui.viz_mesh, 1);
   renderer.updateMeshColor(tetrahedron_id, green);

   data_triangleMesh_t sphere_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );

   int query_point_id = renderer.addMesh(sphere_mesh_data, 0);
   int closest_point_id = renderer.addMesh(sphere_mesh_data, 0);

   renderer.disableGrid();

   renderer.updateMeshColor(query_point_id, red);
   renderer.updateMeshColor(closest_point_id, magenta);

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.setCameraSpeed(0.025f);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0.f, -15.f, 0.f));

   while (true)
   {
      renderer.updateMesh(tetrahedron_id, gui.tetrahedron_ui.viz_mesh);
      renderer.updateMeshTransform(query_point_id, gui.query_point, identityMatrix(), identityMatrix());
      renderer.updateMeshTransform(closest_point_id, gui.closest_bary_pt.point, identityMatrix(), identityMatrix());

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
