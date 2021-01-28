#include "ray_intersection_ui.hpp"

#include "attitudeutils.hpp"
#include "default_camera_controller.hpp"
#include "geometry.hpp"

#include "viz_renderer.hpp"

void MeddlingGui::operator()(void)
{
   ImGui::Begin("Ray intersection test");
   no_window();
   ImGui::End();
}

bool MeddlingGui::no_window(void)
{
   ui_modified_ = false;
   ui_modified_ |= ray.no_window();
   ray.line_type = 1;
   ImGui::Separator();

   ui_modified_ |= triangle.no_window();
   ImGui::Separator();

   ui_modified_ |= aabb.no_window();
   ImGui::Separator();

   ui_modified_ |= poly_loader.no_window();

   if (ui_modified_)
   {
      geometry::types::triangleMesh_t trans_polyhedron(poly_loader.mesh());
      geometry::mesh::applyTransformation(poly_loader.trans_B_to_W(), trans_polyhedron);
      geometry::converters::to_pod(trans_polyhedron, &render_mesh_data);
      calculate_intersections();
   }

   return ui_modified_;
}

void MeddlingGui::calculate_intersections(void)
{
   if (ray.slope.magnitude() < 1e-6f)
   {
      aabb_touching = false;
      triangle_touching = false;
      return;
   }

   Vector3 triangle_intersection_point;

   triangle_touching = geometry::triangle::rayIntersect(
      triangle.triangle.verts[0],
      triangle.triangle.verts[1],
      triangle.triangle.verts[2],
      ray.start,
      ray.slope,
      triangle_intersection_point
   );

   if (!triangle_intersection_point.hasNan() && triangle_touching)
   {
      triangle_touch_point = triangle_intersection_point;
   }
   else
   {
      triangle_touch_point[0] = 1e7f;
   }

   Vector3 aabb_intersection_point;

   aabb_touching = geometry::mesh::rayIntersect(
      aabb.aabb, ray.start, ray.slope, aabb_intersection_point
   );

   if (!aabb_intersection_point.hasNan() && aabb_touching)
   {
      aabb_touch_point = aabb_intersection_point;
   }
   else
   {
      aabb_touch_point[0] = 1e7f;
   }

   Vector3 mesh_intersection_point;

   geometry::types::transform_t tranny = poly_loader.trans_B_to_W();

   geometry::types::triangleMesh_t trans_polyhedron(poly_loader.mesh());
   geometry::mesh::applyTransformation(tranny, trans_polyhedron);
   mesh_touching = geometry::mesh::rayIntersect(
      trans_polyhedron, ray.start, ray.slope, mesh_intersection_point
   );

   if (mesh_touching)
   {
      mesh_touch_point = mesh_intersection_point;
   }
   else
   {
      mesh_touch_point[0] = 1e7f;
   }

}

int main(void)
{
   MeddlingGui gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::CUBE, 0.1f
      );

   int ray_id = renderer.addSegment(2, gui.ray.line_points, red);

   int triangle_id = renderer.addMesh(gui.triangle.viz_mesh, cyan, 0);

   int triangle_touch_id = renderer.addMesh(temp_mesh_data, magenta, 0);

   int aabb_id = renderer.addMesh(gui.aabb.viz_mesh, cyan, 0);

   int aabb_touch_id = renderer.addMesh(temp_mesh_data, magenta, 0);

   int mesh_id = renderer.addMesh(gui.render_mesh_data, cyan, 0);

   while (true)
   {
      if (gui.triangle_touching)
      {
         renderer.updateMeshColor(triangle_id, red);
      }
      else
      {
         renderer.updateMeshColor(triangle_id, cyan);
      }

      if (gui.aabb_touching)
      {
         renderer.updateMeshColor(aabb_id, red);
      }
      else
      {
         renderer.updateMeshColor(aabb_id, cyan);
      }

      if (gui.mesh_touching)
      {
         renderer.updateMeshColor(mesh_id, red);
      }
      else
      {
         renderer.updateMeshColor(mesh_id, cyan);
      }

      if (gui.ui_modified())
      {
         renderer.updateMesh(mesh_id, gui.render_mesh_data);
      }

      renderer.updateMesh(aabb_id, gui.aabb.viz_mesh);
      renderer.updateMesh(triangle_id, gui.triangle.viz_mesh);
      renderer.updateSegment(ray_id, 2, gui.ray.line_points);
      renderer.updateMeshTransform(mesh_id, gui.poly_loader.trans_B_to_W().translate, gui.poly_loader.trans_B_to_W().rotate, gui.poly_loader.trans_B_to_W().scale);
      renderer.updateMeshTransform(aabb_touch_id, gui.aabb_touch_point, identityMatrix(), identityMatrix());
      renderer.updateMeshTransform(triangle_touch_id, gui.triangle_touch_point, identityMatrix(), identityMatrix());

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
