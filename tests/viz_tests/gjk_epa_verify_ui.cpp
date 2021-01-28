// This is really just a visual verification that the MD hull calculation for
// the robust GJK tests is correct.

#include "attitudeutils.hpp"
#include "data_model_io.h"
#include "default_camera_controller.hpp"
#include "gjk_epa_verify_ui.hpp"
#include "matrix33.hpp"
#include "viz_renderer.hpp"

#include <vector>

void MdHullGui::operator()(void)
{
   ImGui::Begin("Yet another GJK tester");
   no_window();
   ImGui::End();
}

bool MdHullGui::no_window(void)
{
   ui_modified_ = false;
   ImGui::Text("Polyhedron A");
   ui_modified_ |= poly_loaders[0].no_window();

   ImGui::Separator();

   ImGui::Text("Polyhedron B");
   ui_modified_ |= poly_loaders[1].no_window();

   ImGui::InputText("Load JSON file", file_path_, 256);
   ImGui::SameLine();
   bool loaded_file = false;
   if (ImGui::ColorButton("file_load_name_", button_color_))
   {
      if (!load_file())
      {
         button_color_.x = 1.f;
         button_color_.y = 0.f;
      }
      else
      {
         button_color_.x = 0.f;
         button_color_.y = 1.f;
         loaded_file = true;
      }
   }
   ImGui::SameLine();
   ImGui::Text("Load mesh");

   ui_modified_ |= loaded_file;

   if (ui_modified_)
   {
      for (int i = 0; i < 2; ++i)
      {
         update_render_mesh(i);
      }

      calculateMdHull();
   }

   return ui_modified_;
}

void MdHullGui::calculateMdHull(void)
{
   geometry::types::triangleMesh_t mesh_A = poly_loaders[0].mesh();
   geometry::types::triangleMesh_t mesh_B = poly_loaders[1].mesh();

   geometry::mesh::applyTransformation(
      poly_loaders[0].trans_B_to_W(), mesh_A
   );
   geometry::mesh::applyTransformation(
      poly_loaders[1].trans_B_to_W(), mesh_B
   );

   geometry::types::triangleMesh_t md_hull;
   int result = test_utils::generateMdHull(mesh_A, mesh_B, md_hull);

   if (result < 0)
   {
      std::cout << "quickhull ran out of triangles or vertices\n";
      return;
   }

   Vector3 zero;
   intersection = geometry::mesh::pointInsideMesh(md_hull, zero);

   geometry::converters::to_pod(md_hull, &md_hull_data);
}

bool MdHullGui::load_file(void)
{
   data_testGjkInput_t test_input_data;
   int load_result = read_data_from_file(&test_input_data, file_path_);

   if (!load_result)
   {
      return false;
   }

   geometry::types::testGjkInput_t test_input;
   geometry::converters::from_pod(&test_input_data, test_input);

   poly_loaders[0].load(test_input.transformA, test_input.shapeA.shapeType);
   poly_loaders[1].load(test_input.transformB, test_input.shapeB.shapeType);

   for (int i = 0; i < 2; ++i)
   {
      if (poly_loaders[i].polyhedron_changed())
      {
         update_render_mesh(i);
      }
   }

   return true;
}

int main(void)
{
   MdHullGui gui;
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::VizRenderer renderer;

   viz::Camera camera;

   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   viz::types::vec4_t blue = {0.f, 0.f, 1.f, 1.f};
   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};
   viz::types::vec4_t yellow = {1.f, 1.f, 0.f, 1.f};

   data_triangleMesh_t & poly_a_data = gui.render_meshes_data[0];
   data_triangleMesh_t & poly_b_data = gui.render_meshes_data[1];
   data_triangleMesh_t & md_hull_data = gui.md_hull_data;

   data_triangleMesh_t temp_cube_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::CUBE, 0.1f
      );

   int poly_a_id = renderer.addMesh(poly_a_data, green, 1);
   int poly_b_id = renderer.addMesh(poly_b_data, red, 1);
   int md_hull_id = renderer.addMesh(md_hull_data, cyan, 0);
   int origin_id = renderer.addMesh(temp_cube_data, magenta, 0);

   geometry::types::triangleMesh_t mesh_A;
   geometry::converters::from_pod(&(poly_a_data), mesh_A);
   geometry::types::triangleMesh_t mesh_B;
   geometry::converters::from_pod(&(poly_b_data), mesh_B);

   // For rendering cubes on top of the Minkowski difference of all the
   // vertices onto the screen.
   std::vector<int> debug_vert_ids;
   debug_vert_ids.reserve(mesh_A.numVerts * mesh_B.numVerts);
   std::vector<Vector3> md_verts_debug;
   md_verts_debug.reserve(mesh_A.numVerts * mesh_B.numVerts);

   // Uncomment to render all MD verts (this and the block in the while loop).
   // for (int i = 0; i < mesh_A.numVerts * mesh_B.numVerts; ++i)
   // {
   //    int debug_vert_id = renderer.addMesh(temp_cube, magenta, 0);
   //    debug_vert_ids.push_back(debug_vert_id);
   // }

   while (true)
   {
      md_verts_debug.clear();

      // Only update mesh vertices if the mesh type has changed.
      if (gui.ui_modified() || gui.poly_loaders[0].polyhedron_changed())
      {
         renderer.updateMesh(poly_a_id, poly_a_data, green);
         geometry::converters::from_pod(&(poly_a_data), mesh_A);
      }

      if (gui.ui_modified() || gui.poly_loaders[1].polyhedron_changed())
      {
         renderer.updateMesh(poly_b_id, poly_b_data, red);
         geometry::converters::from_pod(&(poly_b_data), mesh_B);
      }

      if (gui.intersection)
      {
         renderer.updateMesh(md_hull_id, md_hull_data, magenta);
      }
      else
      {
         renderer.updateMesh(md_hull_id, md_hull_data, cyan);
      }

      renderer.updateMeshTransform(
         poly_a_id,
         gui.poly_loaders[0].trans_B_to_W().translate,
         gui.poly_loaders[0].trans_B_to_W().rotate,
         gui.poly_loaders[0].trans_B_to_W().scale
      );
      renderer.updateMeshTransform(
         poly_b_id,
         gui.poly_loaders[1].trans_B_to_W().translate,
         gui.poly_loaders[1].trans_B_to_W().rotate,
         gui.poly_loaders[1].trans_B_to_W().scale
      );

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
