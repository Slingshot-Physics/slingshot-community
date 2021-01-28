#include "shape_polyhedron_collision_ui.hpp"

#include "collision_contacts.hpp"
#include "epa.hpp"
#include "gjk.hpp"

#include "default_camera_controller.hpp"
#include "viz_renderer.hpp"

bool ImplicitShapeCollision::no_window(void)
{
   ui_modified_ = false;
   ImGui::Checkbox("Show GJK", &show_gjk_);

   ImGui::Checkbox("Show EPA", &show_epa_);

   ImGui::Checkbox("Show contact manifold", &show_contacts_);

   bool shape_loader_modified = shape_loader.no_window();

   bool poly_loader_modified = poly_loader.no_window();

   ui_modified_ |= shape_loader_modified || poly_loader_modified;

   bool ui_changed = shape_loader_modified || poly_loader_modified;

   if (ui_changed)
   {
      update_render_meshes();
      const geometry::types::shape_t & shape = shape_loader.shape();
      const geometry::types::convexPolyhedron_t & poly = poly_loader.polyhedron();

      contacts_W.clear();

      switch(shape_loader.shape().shapeType)
      {
         case geometry::types::enumShape_t::SPHERE:
         {
            gjk_out_ = geometry::gjk::alg(
               shape_loader.trans_B_to_W(),
               poly_loader.trans_B_to_W(),
               shape.sphere,
               poly
            );

            if (gjk_out_.intersection)
            {
               geometry::types::minkowskiDiffSimplex_t tetra_simplex = gjk_out_.minSimplex;

               geometry::epa::expandGjkSimplex(
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  shape.sphere,
                  poly,
                  tetra_simplex
               );

               epa_out_ = geometry::epa::alg(
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  tetra_simplex,
                  shape.sphere,
                  poly
               );

               Vector3 collision_W = rescaleEpaNormal();

               oy::types::contactGeometry_t contact_geom;
               contact_geom.initialize(epa_out_);

               oy::types::collisionContactManifold_t mani = oy::collision::calculateContactManifold(
                  shape.sphere,
                  poly_loader.gauss_map(),
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  contact_geom
               );

               for (unsigned int i = 0; i < mani.numContacts; ++i)
               {
                  contacts_W.push_back(mani.bodyAContacts[i]);
               }
            }

            break;
         }
         case geometry::types::enumShape_t::CAPSULE:
         {
            gjk_out_ = geometry::gjk::alg(
               shape_loader.trans_B_to_W(),
               poly_loader.trans_B_to_W(),
               shape.capsule,
               poly
            );

            if (gjk_out_.intersection)
            {
               geometry::types::minkowskiDiffSimplex_t tetra_simplex = gjk_out_.minSimplex;

               geometry::epa::expandGjkSimplex(
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  shape.capsule,
                  poly,
                  tetra_simplex
               );

               epa_out_ = geometry::epa::alg(
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  tetra_simplex,
                  shape.capsule,
                  poly
               );

               Vector3 collision_W = rescaleEpaNormal();

               oy::types::contactGeometry_t contact_geom;
               contact_geom.initialize(epa_out_);

               oy::types::collisionContactManifold_t mani = oy::collision::calculateContactManifold(
                  shape.capsule,
                  poly_loader.gauss_map(),
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  contact_geom
               );

               for (unsigned int i = 0; i < mani.numContacts; ++i)
               {
                  contacts_W.push_back(mani.bodyAContacts[i]);
               }
            }

            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            gjk_out_ = geometry::gjk::alg(
               shape_loader.trans_B_to_W(),
               poly_loader.trans_B_to_W(),
               shape.cylinder,
               poly
            );

            if (gjk_out_.intersection)
            {
               geometry::types::minkowskiDiffSimplex_t tetra_simplex = gjk_out_.minSimplex;

               geometry::epa::expandGjkSimplex(
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  shape.cylinder,
                  poly,
                  tetra_simplex
               );

               epa_out_ = geometry::epa::alg(
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  tetra_simplex,
                  shape.cylinder,
                  poly
               );

               Vector3 collision_W = rescaleEpaNormal();

               oy::types::contactGeometry_t contact_geom;
               contact_geom.initialize(epa_out_);

               oy::types::collisionContactManifold_t mani = oy::collision::calculateContactManifold(
                  shape.cylinder,
                  poly_loader.gauss_map(),
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  contact_geom
               );

               for (unsigned int i = 0; i < mani.numContacts; ++i)
               {
                  contacts_W.push_back(mani.bodyAContacts[i]);
               }
            }

            break;
         }
         case geometry::types::enumShape_t::CUBE:
         {
            gjk_out_ = geometry::gjk::alg(
               shape_loader.trans_B_to_W(),
               poly_loader.trans_B_to_W(),
               shape.cube,
               poly
            );

            if (gjk_out_.intersection)
            {
               geometry::types::minkowskiDiffSimplex_t tetra_simplex = gjk_out_.minSimplex;

               geometry::epa::expandGjkSimplex(
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  shape.cube,
                  poly,
                  tetra_simplex
               );

               epa_out_ = geometry::epa::alg(
                  shape_loader.trans_B_to_W(),
                  poly_loader.trans_B_to_W(),
                  tetra_simplex,
                  shape.cube,
                  poly
               );
            }

            break;
         }
         default:
            break;
      }

      gjk_point_a_W.Initialize(0.f, 0.f, 0.f);
      gjk_point_b_W.Initialize(0.f, 0.f, 0.f);
      for (int i = 0; i < 4; ++i)
      {
         gjk_point_a_W += gjk_out_.minSimplex.minNormBary[i] * geometry::transform::forwardBound(
            shape_loader.trans_B_to_W(), gjk_out_.minSimplex.bodyAVerts[i]
         );
         gjk_point_b_W += gjk_out_.minSimplex.minNormBary[i] * geometry::transform::forwardBound(
            poly_loader.trans_B_to_W(), gjk_out_.minSimplex.bodyBVerts[i]
         );
      }

      if (gjk_out_.intersection && epa_out_.collided)
      {
         epa_contact_point_a_W = epa_out_.bodyAContactPoint;
         epa_contact_point_b_W = epa_out_.bodyBContactPoint;
         epa_collision_normal_W = epa_out_.p;
         epa_collision_normal_W.Normalize();
      }
      else
      {
         epa_contact_point_a_W.Initialize(0.f, 0.f, 0.f);
         epa_contact_point_b_W.Initialize(0.f, 0.f, 0.f);
         epa_collision_normal_W.Initialize(0.f, 0.f, 0.f);
      }
   }

   return ui_modified_;
}

int main(void)
{
   ImplicitShapeCollision gui;
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

   int shape_mesh_id = renderer.addMesh(gui.shape_render_data, red, 0);

   int poly_mesh_id = renderer.addMesh(gui.polyhedron_render_data, green, 1);

   data_triangleMesh_t temp_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::CUBE, 0.1f
      );

   int gjk_shape_point_id = renderer.addMesh(temp_mesh_data, cyan, 0);

   int gjk_poly_point_id = renderer.addMesh(temp_mesh_data, cyan, 0);

   int epa_contact_pt_a_id = renderer.addMesh(temp_mesh_data, purple, 0);

   int epa_contact_pt_b_id = renderer.addMesh(temp_mesh_data, purple, 0);

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
      if (gui.ui_modified())
      {
         renderer.updateMesh(shape_mesh_id, gui.shape_render_data);
      }

      if (gui.ui_modified())
      {
         renderer.updateMesh(poly_mesh_id, gui.polyhedron_render_data);
      }

      renderer.updateMeshTransform(
         shape_mesh_id,
         gui.shape_loader.trans_B_to_W().translate,
         gui.shape_loader.trans_B_to_W().rotate,
         gui.shape_loader.trans_B_to_W().scale
      );

      renderer.updateMeshTransform(
         poly_mesh_id,
         gui.poly_loader.trans_B_to_W().translate,
         gui.poly_loader.trans_B_to_W().rotate,
         gui.poly_loader.trans_B_to_W().scale
      );

      renderer.updateMeshTransform(
         gjk_shape_point_id,
         gui.gjk_point_a_W,
         identityMatrix(),
         identityMatrix()
      );

      renderer.updateMeshTransform(
         gjk_poly_point_id,
         gui.gjk_point_b_W,
         identityMatrix(),
         identityMatrix()
      );

      if (gui.gjk_out().intersection)
      {
         renderer.updateMeshColor(gjk_shape_point_id, orange);
         renderer.updateMeshColor(gjk_poly_point_id, orange);
      }
      else
      {
         renderer.updateMeshColor(gjk_shape_point_id, cyan);
         renderer.updateMeshColor(gjk_poly_point_id, cyan);
      }

      if (gui.show_gjk())
      {
         renderer.enableMesh(gjk_shape_point_id);
         renderer.enableMesh(gjk_poly_point_id);
      }
      else
      {
         renderer.disableMesh(gjk_shape_point_id);
         renderer.disableMesh(gjk_poly_point_id);
      }

      if (gui.show_epa())
      {
         if (gui.gjk_out().intersection && gui.epa_out().collided)
         {
            collision_normal_segment[0] = gui.epa_contact_point_a_W;
            collision_normal_segment[1] = gui.epa_contact_point_a_W + gui.epa_collision_normal_W;
            renderer.enableMesh(epa_contact_pt_a_id);
            renderer.enableMesh(epa_contact_pt_b_id);
            renderer.enableMesh(collision_normal_id);
         }
         else
         {
            renderer.disableMesh(epa_contact_pt_a_id);
            renderer.disableMesh(epa_contact_pt_b_id);
            renderer.disableMesh(collision_normal_id);
         }
      }
      else
      {
         renderer.disableMesh(epa_contact_pt_a_id);
         renderer.disableMesh(epa_contact_pt_b_id);
         renderer.disableMesh(collision_normal_id);
      }

      if (gui.gjk_out().intersection && gui.epa_out().collided)
      {
         for (int i = 0; i < 50; ++i)
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
      }

      renderer.updateSegment(collision_normal_id, 2, collision_normal_segment);
      renderer.updateMeshTransform(epa_contact_pt_a_id, gui.epa_contact_point_a_W, identityMatrix(), identityMatrix());
      renderer.updateMeshTransform(epa_contact_pt_b_id, gui.epa_contact_point_b_W, identityMatrix(), identityMatrix());

      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
