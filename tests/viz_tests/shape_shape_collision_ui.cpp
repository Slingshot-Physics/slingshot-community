#include "shape_shape_collision_ui.hpp"

#include "data_model_io.h"
#include "default_camera_controller.hpp"
#include "viz_renderer.hpp"

void sphere_sphere(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.sphere,
      shape_b.sphere,
      gjk_out,
      epa_out,
      contacts
   );
}

void sphere_capsule(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.sphere,
      shape_b.capsule,
      gjk_out,
      epa_out,
      contacts
   );
}

void sphere_cylinder(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.sphere,
      shape_b.cylinder,
      gjk_out,
      epa_out,
      contacts
   );
}

void capsule_sphere(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.capsule,
      shape_b.sphere,
      gjk_out,
      epa_out,
      contacts
   );
}

void capsule_capsule(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.capsule,
      shape_b.capsule,
      gjk_out,
      epa_out,
      contacts
   );
}

void capsule_cylinder(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.capsule,
      shape_b.cylinder,
      gjk_out,
      epa_out,
      contacts
   );
}

void cylinder_sphere(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.cylinder,
      shape_b.sphere,
      gjk_out,
      epa_out,
      contacts
   );
}

void cylinder_capsule(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.cylinder,
      shape_b.capsule,
      gjk_out,
      epa_out,
      contacts
   );
}

void cylinder_cylinder(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   gjk_epa_manifold(
      trans_A_to_W,
      trans_B_to_W,
      shape_a.cylinder,
      shape_b.cylinder,
      gjk_out,
      epa_out,
      contacts
   );
}

bool ShapeShapeCollision::no_window(void)
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
   }

   ImGui::Separator();

   ImGui::InputText("Load JSON file", &file_path_);
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

   return ui_modified_;
}

void ShapeShapeCollision::collision_multiplexor(void)
{
   geometry::types::shape_t shape_a = shape_loaders[0].shape();
   geometry::types::shape_t shape_b = shape_loaders[1].shape();

   geometry::types::transform_t trans_A_to_W = shape_loaders[0].trans_B_to_W();
   geometry::types::transform_t trans_B_to_W = shape_loaders[1].trans_B_to_W();

   ShapeShapeCollision_f collision_func = call_matrix_[static_cast<int>(shape_a.shapeType)][static_cast<int>(shape_b.shapeType)];

   if (collision_func == nullptr)
   {
      std::cout << "couldn't load collision function for shapes: " << static_cast<int>(shape_a.shapeType) << ", " << static_cast<int>(shape_b.shapeType) << "\n";
      return;
   }

   collision_func(
      trans_A_to_W, trans_B_to_W, shape_a, shape_b, gjk_out_, epa_out_, contacts_W
   );

   gjk_points_W[0].Initialize(0.f, 0.f, 0.f);
   gjk_points_W[1].Initialize(0.f, 0.f, 0.f);

   for (int i = 0; i < gjk_out_.minSimplex.numVerts; ++i)
   {
      gjk_points_W[0] += gjk_out_.minSimplex.minNormBary[i] * geometry::transform::forwardBound(
         shape_loaders[0].trans_B_to_W(), gjk_out_.minSimplex.bodyAVerts[i]
      );
      gjk_points_W[1] += gjk_out_.minSimplex.minNormBary[i] * geometry::transform::forwardBound(
         shape_loaders[1].trans_B_to_W(), gjk_out_.minSimplex.bodyBVerts[i]
      );
   }

   if (gjk_out_.intersection && epa_out_.collided)
   {
      epa_contact_points_W[0] = epa_out_.bodyAContactPoint;
      epa_contact_points_W[1] = epa_out_.bodyBContactPoint;
      epa_collision_normal_W = epa_out_.p;
      epa_collision_normal_W.Normalize();
   }
   else
   {
      epa_contact_points_W[0].Initialize(0.f, 0.f, 0.f);
      epa_contact_points_W[1].Initialize(0.f, 0.f, 0.f);
      epa_collision_normal_W.Initialize(0.f, 0.f, 0.f);
   }
}

bool ShapeShapeCollision::load_file(void)
{
   data_testGjkInput_t test_input_data;
   int load_result = read_data_from_file(&test_input_data, file_path_.c_str());

   if (!load_result)
   {
      return false;
   }

   geometry::types::testGjkInput_t test_input;
   geometry::converters::from_pod(&test_input_data, test_input);

   shape_loaders[0].load(test_input.transformA, test_input.shapeA);
   shape_loaders[1].load(test_input.transformB, test_input.shapeB);

   update_render_meshes();

   return true;
}

int main(void)
{
   ShapeShapeCollision gui;
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

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::CUBE, 0.1f
      );

   int shape_a_aabb_id = renderer.addMesh(gui.shape_a_aabb_mesh_data, yellow, 1);

   int shape_mesh_ids[2] = {
      renderer.addMesh(gui.shape_render_data[0], red, 1),
      renderer.addMesh(gui.shape_render_data[1], green, 1)
   };

   int gjk_point_ids[2] = {
      renderer.addMesh(temp_mesh_data, cyan, 0),
      renderer.addMesh(temp_mesh_data, cyan, 0)
   };

   int epa_contact_pt_ids[2] = {
      renderer.addMesh(temp_mesh_data, purple, 0),
      renderer.addMesh(temp_mesh_data, purple, 0),
   };

   std::vector<int> contact_ids;
   for (int i = 0; i < 50; ++i)
   {
      contact_ids.push_back(renderer.addMesh(temp_mesh_data, blue, 0));
      renderer.disableMesh(contact_ids[i]);
   }

   Vector3 collision_normal_segment[2] = {
      {0.f, 0.f, 0.f},
      {0.f, 0.f, 0.f}
   };

   int collision_normal_id = renderer.addSegment(2, collision_normal_segment, yellow);

   while (true)
   {
      for (int i = 0; i < 2; ++i)
      {
         if (gui.ui_modified())
         {
            renderer.updateMesh(shape_mesh_ids[i], gui.shape_render_data[i]);
            renderer.updateMesh(shape_a_aabb_id, gui.shape_a_aabb_mesh_data);

            geometry::types::transform_t trans_M_to_W = gui.shape_loaders[i].trans_B_to_W();
            renderer.updateMeshTransform(shape_mesh_ids[i], trans_M_to_W.translate, trans_M_to_W.rotate, trans_M_to_W.scale);
            renderer.updateMeshTransform(gjk_point_ids[i], gui.gjk_points_W[i], identityMatrix(), identityMatrix());
            renderer.updateMeshTransform(epa_contact_pt_ids[i], gui.epa_contact_points_W[i], identityMatrix(), identityMatrix());
            collision_normal_segment[0] = gui.epa_contact_points_W[0];
            collision_normal_segment[1] = gui.epa_contact_points_W[0] + gui.epa_collision_normal_W;
            renderer.updateSegment(collision_normal_id, 2, collision_normal_segment);
         }

         if (gui.gjk_out().intersection)
         {
            renderer.updateMeshColor(gjk_point_ids[i], orange);
         }
         else
         {
            renderer.updateMeshColor(gjk_point_ids[i], cyan);
         }

         if (gui.epa_out().collided && gui.gjk_out().intersection)
         {
            renderer.enableMesh(epa_contact_pt_ids[i]);
         }
         else
         {
            renderer.disableMesh(epa_contact_pt_ids[i]);
         }
      }

      for (unsigned int i = 0; i < contact_ids.size(); ++i)
      {
         if (i < gui.contacts_W.size())
         {
            renderer.enableMesh(contact_ids[i]);
            renderer.updateMeshTransform(contact_ids[i], gui.contacts_W[i], identityMatrix(), identityMatrix());
         }
         else
         {
            renderer.disableMesh(contact_ids[i]);
         }
      }

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
