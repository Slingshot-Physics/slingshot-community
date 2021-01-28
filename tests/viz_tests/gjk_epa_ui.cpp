#include <algorithm>
#include <iostream>

#include "attitudeutils.hpp"
#include "collision_contacts.hpp"
#include "default_camera_controller.hpp"
#include "data_model_io.h"
#include "epa.hpp"
#include "slingshot_types.hpp"
#include "gauss_map.hpp"
#include "geometry_types.hpp"
#include "geometry_type_converters.hpp"
#include "gjk_epa_ui.hpp"
#include "gjk.hpp"
#include "logger_utils.hpp"
#include "viz_renderer.hpp"

void MeddlingGui::operator()(void)
{
   ImGui::Begin("GJK tester");

   no_window();

   ImGui::End();
}

bool MeddlingGui::no_window(void)
{
   ui_modified_ = false;

   ui_modified_ |= ImGui::Checkbox("Run GJK", &run_gjk);
   ImGui::SameLine();
   ui_modified_ |= ImGui::Checkbox("Show GJK closest points", &show_gjk_points);

   show_gjk_points &= run_gjk;

   ImGui::Separator();

   bool epa_changed = ImGui::Checkbox("Run EPA", &run_epa);
   ui_modified_ |= epa_changed;
   ImGui::SameLine();
   ui_modified_ |= ImGui::Checkbox("Show EPA contact points", &show_epa_points);
   ImGui::SameLine();
   ui_modified_ |= ImGui::Checkbox("Show EPA collision normal", &show_collision_normal);

   run_epa = (run_gjk && run_epa);
   show_epa_points &= run_epa;
   show_collision_normal &= run_epa;

   ImGui::Separator();

   bool contact_manifold_changed = ImGui::Checkbox("Run contact manifold", &run_manifold);
   ui_modified_ |= contact_manifold_changed;
   ImGui::SameLine();
   ui_modified_ |= ImGui::Checkbox("Show contact manifold", &show_manifold);

   run_manifold = (run_manifold & run_epa);
   show_manifold &= run_manifold;

   ImGui::Separator();

   bool poly_loader_modified[2] = {false, false};
   ImGui::Text("Polyhedron A");
   poly_loader_modified[0] |= poly_loaders[0].no_window();
   ui_modified_ |= poly_loader_modified[0];

   ImGui::Separator();

   ImGui::Text("Polyhedron B");
   poly_loader_modified[1] |= poly_loaders[1].no_window();
   ui_modified_ |= poly_loader_modified[1];

   ImGui::Separator();

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
   ui_modified_ |= loaded_file;
   ImGui::SameLine();
   ImGui::Text("Load mesh");

   if(ImGui::Button("Save bad states"))
   {
      // save_file(false);
   }
   ImGui::SameLine();
   if(ImGui::Button("Save good states"))
   {
      // save_file(true);
   }

   bool input_changed = (
      loaded_file ||
      poly_loader_modified[0] ||
      poly_loader_modified[1]
   );

   for (int i = 0; i < 2; ++i)
   {
      if (poly_loaders[i].polyhedron_changed() || poly_loader_modified[i])
      {
         update_render_mesh(i);
      }
   }

   if (run_gjk && input_changed)
   {
      run_gjk_alg();
   }

   if (
      (run_epa && gjk_out.intersection && input_changed) ||
      (run_epa && gjk_out.intersection && epa_changed)
   )
   {
      run_epa_alg();
   }

   if (
      (gjk_out.intersection && epa_out.collided && run_manifold && input_changed) ||
      (gjk_out.intersection && epa_out.collided && run_manifold && contact_manifold_changed)
   )
   {
      std::cout << "running manifold alg\n";
      run_contact_manifold_alg();
   }

   return ui_modified_;
}

bool MeddlingGui::load_file(void)
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

void MeddlingGui::run_gjk_alg(void)
{
   const geometry::types::transform_t & trans_A_to_W = poly_loaders[0].trans_B_to_W();
   const geometry::types::transform_t & trans_B_to_W = poly_loaders[1].trans_B_to_W();

   gjk_out = geometry::gjk::alg(
      trans_A_to_W,
      trans_B_to_W,
      poly_loaders[0].polyhedron(),
      poly_loaders[1].polyhedron()
   );

   closest_pt_a_world.Initialize(0.f, 0.f, 0.f);
   closest_pt_b_world.Initialize(0.f, 0.f, 0.f);

   for (int i = 0; i < gjk_out.minSimplex.numVerts; ++i)
   {
      closest_pt_a_world += gjk_out.minSimplex.minNormBary[i] * geometry::transform::forwardBound(trans_A_to_W, gjk_out.minSimplex.bodyAVerts[i]);
      closest_pt_b_world += gjk_out.minSimplex.minNormBary[i] * geometry::transform::forwardBound(trans_B_to_W, gjk_out.minSimplex.bodyBVerts[i]);
   }
}

void MeddlingGui::run_epa_alg(void)
{
   const geometry::types::transform_t & trans_A_to_W = poly_loaders[0].trans_B_to_W();
   const geometry::types::transform_t & trans_B_to_W = poly_loaders[1].trans_B_to_W();

   geometry::types::minkowskiDiffSimplex_t tetra_simplex = gjk_out.minSimplex;

   geometry::epa::expandGjkSimplex(
      poly_loaders[0].trans_B_to_W(),
      poly_loaders[1].trans_B_to_W(),
      poly_loaders[0].polyhedron(),
      poly_loaders[1].polyhedron(),
      tetra_simplex
   );

   epa_out = geometry::epa::alg(
      trans_A_to_W,
      trans_B_to_W,
      tetra_simplex,
      poly_loaders[0].polyhedron(),
      poly_loaders[1].polyhedron()
   );

   mesh_md_W = geometry::epa::alg_debug(
      trans_A_to_W,
      trans_B_to_W,
      tetra_simplex,
      poly_loaders[0].polyhedron(),
      poly_loaders[1].polyhedron()
   );

   // mesh_md_W.print();

   mesh_md_W.to_triangle_mesh(&epa_mesh_data);

   if (epa_out.collided)
   {
      contact_pt_a_world = epa_out.bodyAContactPoint;
      contact_pt_b_world = epa_out.bodyBContactPoint;

      collision_normal_world = epa_out.p;
      collision_normal_world.Normalize();
   }
}

void MeddlingGui::run_contact_manifold_alg(void)
{
   Vector3 poly_a_pos = poly_loaders[0].trans_B_to_W().translate;
   Vector3 poly_b_pos = poly_loaders[1].trans_B_to_W().translate;

   Vector3 md_cm_vec_W = poly_b_pos - poly_a_pos;
   float sign = -1.f + (md_cm_vec_W.dot(epa_out.p) > 0.f) * 2.f;

   Vector3 collision_W = (sign * epa_out.p).unitVector();

   contacts.clear();

   oy::types::contactGeometry_t contact_geom;
   contact_geom.initialize(epa_out);

   oy::types::collisionContactManifold_t mani = oy::collision::calculateContactManifold(
      poly_loaders[0].gauss_map(),
      poly_loaders[1].gauss_map(),
      poly_loaders[0].trans_B_to_W(),
      poly_loaders[1].trans_B_to_W(),
      contact_geom
   );

   for (unsigned int i = 0; i < mani.numContacts; ++i)
   {
      contacts.push_back(mani.bodyAContacts[i]);
   }
}

int main(void)
{
   MeddlingGui gui;
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

   data_triangleMesh_t & poly_a_data = gui.render_meshes_data[0];
   data_triangleMesh_t & poly_b_data = gui.render_meshes_data[1];

   int poly_a_id = renderer.addMesh(poly_a_data, green, 0);
   int poly_b_id = renderer.addMesh(poly_b_data, red, 1);

   data_triangleMesh_t cube_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::CUBE, 0.1f
      );

   int closest_pt_a_id = renderer.addMesh(cube_data, magenta, 0);
   int closest_pt_b_id = renderer.addMesh(cube_data, magenta, 0);

   renderer.disableMesh(closest_pt_a_id);
   renderer.disableMesh(closest_pt_b_id);

   int contact_pt_a_id = renderer.addMesh(cube_data, purple, 0);
   int contact_pt_b_id = renderer.addMesh(cube_data, purple, 0);

   renderer.disableMesh(contact_pt_a_id);
   renderer.disableMesh(contact_pt_b_id);

   int epa_md_mesh_id = renderer.addMesh(cube_data, magenta, 0);

   renderer.disableMesh(epa_md_mesh_id);

   std::vector<int> contact_ids;
   for (int i = 0; i < 50; ++i)
   {
      contact_ids.push_back(renderer.addMesh(cube_data, blue, 0));
      renderer.disableMesh(contact_ids[i]);
   }

   Vector3 collision_normal_segment[2] = {
      {0.f, 0.f, 0.f},
      {0.f, 0.f, 0.f}
   };

   int collision_normal_id = renderer.addSegment(2, collision_normal_segment, yellow);

   const geometry::types::transform_t & trans_A_to_W = gui.poly_loaders[0].trans_B_to_W();
   const geometry::types::transform_t & trans_B_to_W = gui.poly_loaders[1].trans_B_to_W();

   while (true)
   {
      // This code is here in case a new polyhedron is selected.
      if (gui.ui_modified())
      {
         renderer.updateMesh(poly_a_id, poly_a_data, green);
         renderer.updateMesh(poly_b_id, poly_b_data, red);
      }

      renderer.updateMeshTransform(poly_a_id, trans_A_to_W.translate, trans_A_to_W.rotate, trans_A_to_W.scale);
      renderer.updateMeshTransform(poly_b_id, trans_B_to_W.translate, trans_B_to_W.rotate, trans_B_to_W.scale);

      bool gjk_collision = gui.gjk_out.intersection;

      if (gui.show_gjk_points)
      {
         renderer.enableMesh(closest_pt_a_id);
         renderer.enableMesh(closest_pt_b_id);
         renderer.updateMeshTransform(
            closest_pt_a_id, gui.closest_pt_a_world, identityMatrix(), identityMatrix()
         );

         renderer.updateMeshTransform(
            closest_pt_b_id, gui.closest_pt_b_world, identityMatrix(), identityMatrix()
         );

         if (gjk_collision)
         {
            renderer.updateMeshColor(closest_pt_a_id, orange);
            renderer.updateMeshColor(closest_pt_b_id, orange);
         }
         else
         {
            renderer.updateMeshColor(closest_pt_a_id, cyan);
            renderer.updateMeshColor(closest_pt_b_id, cyan);
         }
      }
      else
      {
         renderer.disableMesh(closest_pt_a_id);
         renderer.disableMesh(closest_pt_b_id);
      }
      bool epa_collision = gui.epa_out.collided;

      if (gjk_collision && gui.ui_modified())
      {
         renderer.updateMesh(epa_md_mesh_id, gui.epa_mesh_data);
         renderer.updateMeshColor(epa_md_mesh_id, magenta);
      }

      if (gjk_collision && gui.show_epa_points)
      {
         renderer.enableMesh(epa_md_mesh_id);
      }
      else
      {
         renderer.disableMesh(epa_md_mesh_id);
      }

      if (gjk_collision && epa_collision && gui.show_epa_points)
      {
         renderer.enableMesh(contact_pt_a_id);
         renderer.enableMesh(contact_pt_b_id);
         renderer.updateMeshTransform(contact_pt_a_id, gui.contact_pt_a_world, identityMatrix(), identityMatrix());
         renderer.updateMeshTransform(contact_pt_b_id, gui.contact_pt_b_world, identityMatrix(), identityMatrix());
      }
      else
      {
         renderer.disableMesh(contact_pt_a_id);
         renderer.disableMesh(contact_pt_b_id);
      }

      if (gjk_collision && epa_collision && gui.show_manifold)
      {
         for (int i = 0; i < 50; ++i)
         {
            if (i < gui.contacts.size())
            {
               renderer.enableMesh(contact_ids[i]);
               renderer.updateMeshTransform(contact_ids[i], gui.contacts[i], identityMatrix(), identityMatrix());
            }
            else
            {
               renderer.disableMesh(contact_ids[i]);
            }
         }
      }
      else
      {
         for (int i = 0; i < 50; ++i)
         {
            renderer.disableMesh(contact_ids[i]);
         }
      }

      if (gjk_collision && epa_collision & gui.show_collision_normal)
      {
         collision_normal_segment[0] = gui.contact_pt_a_world;
         collision_normal_segment[1] = gui.contact_pt_a_world + gui.collision_normal_world;
         renderer.enableMesh(collision_normal_id);
      }
      else
      {
         renderer.disableMesh(collision_normal_id);
      }
      renderer.updateSegment(collision_normal_id, 2, collision_normal_segment);

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
