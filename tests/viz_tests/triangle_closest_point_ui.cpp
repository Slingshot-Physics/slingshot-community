#include "triangle_closest_point_ui.hpp"

#include "mesh.hpp"
#include "triangle.hpp"

#include "default_camera_controller.hpp"
#include "geometry_type_converters.hpp"
#include "viz_renderer.hpp"

void TriangleClosestPointUI::operator()(void)
{
   ImGui::Begin("triangle closest point UI");
   no_window();
   ImGui::End();
}

bool TriangleClosestPointUI::no_window(void)
{
   bool ui_modified = false;
   ui_modified |= triangle_ui.no_window();
   ui_modified |= ImGui::DragFloat3("query point", &(query_point[0]), 0.025f, -32.f, 32.f, "%0.7f");

   ImGui::Text("closest point bary coord: [%f, %f, %f]", closest_bary_pt.bary[0], closest_bary_pt.bary[1], closest_bary_pt.bary[2]);
   ImGui::Text("query point bary coord: [%f, %f, %f]", query_bary_pt[0], query_bary_pt[1], query_bary_pt[2]);

   if (ui_modified)
   {
      query_bary_pt = geometry::triangle::baryCoords(triangle_ui.triangle, query_point);
      closest_bary_pt = geometry::triangle::closestPointToPoint(
         triangle_ui.triangle.verts[0], triangle_ui.triangle.verts[1], triangle_ui.triangle.verts[2], query_point
      );
   }

   return ui_modified;
}

int main(void)
{
   TriangleClosestPointUI gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   int triangle_id = renderer.addSegment(3, gui.triangle_ui.triangle.verts, green);

   data_triangleMesh_t sphere_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::SPHERE, 0.1f
      );

   int query_point_id = renderer.addMesh(sphere_mesh_data, 0);
   int closest_point_id = renderer.addMesh(sphere_mesh_data, 0);

   renderer.updateMeshColor(query_point_id, red);
   renderer.updateMeshColor(closest_point_id, magenta);

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0.f, -15.f, 0.f));

   while (true)
   {
      renderer.updateSegment(triangle_id, 3, gui.triangle_ui.triangle.verts);
      renderer.updateMeshTransform(query_point_id, gui.query_point, identityMatrix(), identityMatrix());
      renderer.updateMeshTransform(closest_point_id, gui.closest_bary_pt.point, identityMatrix(), identityMatrix());

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
