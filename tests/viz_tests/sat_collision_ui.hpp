#ifndef SAT_UI_HEADER
#define SAT_UI_HEADER

#include "gui_callback_base.hpp"

#include "sat.hpp"

#include "aabb_hull.hpp"
#include "collision_contacts.hpp"
#include "slingshot_types.hpp"
#include "geometry_types.hpp"
#include "geometry_type_converters.hpp"

#include "shape_loader_ui.hpp"

class SatCollision : public viz::GuiCallbackBase
{
   public:
      SatCollision(void)
         : shape_loaders{
            {"shape A", {0.f, 0.f, 2.f}},
            {"shape B", {0.f, 0.f, -2.f}}
         }
         , ui_modified_(false)
         , sat_manifold_(true)
      {
         contact_manifold.numContacts = 0;
         update_aabb_mesh();
         update_render_meshes();
      }

      void operator()(void)
      {
         ImGui::Begin("SAT Collision UI");

         no_window();

         ImGui::End();
      }

      bool no_window(void);

      bool ui_modified(void) const
      {
         return ui_modified_;
      }

      const geometry::types::satResult_t sat_out(void) const
      {
         return sat_out_;
      }

      ShapeLoader shape_loaders[2];

      data_triangleMesh_t shape_render_data[2];

      Vector3 collision_normal_W;

      oy::types::collisionContactManifold_t contact_manifold;

      data_triangleMesh_t shape_a_aabb_mesh_data;

   private:

      bool ui_modified_;

      geometry::types::satResult_t sat_out_;

      bool sat_manifold_;

      void update_render_meshes(void)
      {
         geometry::types::triangleMesh_t temp_mesh;
         for (int i = 0; i < 2; ++i)
         {
            temp_mesh = shape_loaders[i].mesh();
            geometry::converters::to_pod(temp_mesh, &(shape_render_data[i]));
         }
      }

      void collision_multiplexor(void)
      {
         geometry::types::shape_t shape_a = shape_loaders[0].shape();
         geometry::types::shape_t shape_b = shape_loaders[1].shape();

         geometry::types::transform_t trans_A_to_W = shape_loaders[0].trans_B_to_W();
         geometry::types::transform_t trans_B_to_W = shape_loaders[1].trans_B_to_W();

         geometry::types::isometricTransform_t isom_trans_A_to_W = {trans_A_to_W.rotate, trans_A_to_W.translate};
         geometry::types::isometricTransform_t isom_trans_B_to_W = {trans_B_to_W.rotate, trans_B_to_W.translate};

         if (
            (shape_a.shapeType == geometry::types::enumShape_t::CUBE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CUBE)
         )
         {
            sat_out_ = geometry::collisions::cubeCube(
               isom_trans_A_to_W,
               isom_trans_B_to_W,
               shape_a.cube,
               shape_b.cube
            );

            if (sat_out_.collision)
            {
               contact_manifold_alg(
                  isom_trans_A_to_W,
                  isom_trans_B_to_W,
                  shape_a.cube,
                  shape_b.cube
               );
            }
         }
         else if(
            (shape_a.shapeType == geometry::types::enumShape_t::CUBE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::SPHERE)
         )
         {
            sat_out_ = geometry::collisions::cubeSphere(
               isom_trans_A_to_W,
               isom_trans_B_to_W,
               shape_a.cube,
               shape_b.sphere
            );

            if (sat_out_.collision)
            {
               contact_manifold_alg(
                  isom_trans_A_to_W,
                  isom_trans_B_to_W,
                  shape_a.cube,
                  shape_b.sphere
               );
            }
         }
         else if(
            (shape_a.shapeType == geometry::types::enumShape_t::CUBE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CAPSULE)
         )
         {
            sat_out_ = geometry::collisions::cubeCapsule(
               isom_trans_A_to_W,
               isom_trans_B_to_W,
               shape_a.cube,
               shape_b.capsule
            );

            if (sat_out_.collision)
            {
               contact_manifold_alg(
                  isom_trans_A_to_W,
                  isom_trans_B_to_W,
                  shape_a.cube,
                  shape_b.capsule
               );
            }
         }
         else if(
            (shape_a.shapeType == geometry::types::enumShape_t::SPHERE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::SPHERE)
         )
         {
            sat_out_ = geometry::collisions::sphereSphere(
               isom_trans_A_to_W,
               isom_trans_B_to_W,
               shape_a.sphere,
               shape_b.sphere
            );

            if (sat_out_.collision)
            {
               contact_manifold_alg(
                  isom_trans_A_to_W,
                  isom_trans_B_to_W,
                  shape_a.sphere,
                  shape_b.sphere
               );
            }
         }
         else if(
            (shape_a.shapeType == geometry::types::enumShape_t::SPHERE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CAPSULE)
         )
         {
            sat_out_ = geometry::collisions::sphereCapsule(
               isom_trans_A_to_W,
               isom_trans_B_to_W,
               shape_a.sphere,
               shape_b.capsule
            );

            if (sat_out_.collision)
            {
               contact_manifold_alg(
                  isom_trans_A_to_W,
                  isom_trans_B_to_W,
                  shape_a.sphere,
                  shape_b.capsule
               );
            }
         }
         else if (
            (shape_a.shapeType == geometry::types::enumShape_t::SPHERE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CYLINDER)
         )
         {
            sat_out_ = geometry::collisions::sphereCylinder(
               isom_trans_A_to_W,
               isom_trans_B_to_W,
               shape_a.sphere,
               shape_b.cylinder
            );

            if (sat_out_.collision)
            {
               contact_manifold_alg(
                  isom_trans_A_to_W,
                  isom_trans_B_to_W,
                  shape_a.sphere,
                  shape_b.cylinder
               );
            }
         }
         else
         {
            sat_out_.collision = false;
            std::cout << "unsupported collision between " << static_cast<int>(shape_a.shapeType) << ", " << static_cast<int>(shape_b.shapeType) << "\n";
         }

         if (sat_manifold_)
         {
            contact_manifold.numContacts = sat_out_.numDeepestPointPairs;
            for (int i = 0; i < sat_out_.numDeepestPointPairs; ++i)
            {
               contact_manifold.bodyAContacts[i] = sat_out_.deepestPointsA[i];
               contact_manifold.bodyBContacts[i] = sat_out_.deepestPointsB[i];
            }
         }

         if (!sat_out_.collision)
         {
            contact_manifold.numContacts = 0;
         }
      }

      template <typename ShapeA_T, typename ShapeB_T>
      void contact_manifold_alg(
         geometry::types::isometricTransform_t isom_trans_A_to_W,
         geometry::types::isometricTransform_t isom_trans_B_to_W,
         const ShapeA_T shape_a,
         const ShapeB_T shape_b
      )
      {
         oy::types::contactGeometry_t temp_con_geom;
         temp_con_geom.contactNormal = sat_out_.contactNormal;
         temp_con_geom.bodyAContactPoint = sat_out_.deepestPointsA[0];
         temp_con_geom.bodyBContactPoint = sat_out_.deepestPointsB[0];

         contact_manifold = oy::collision::calculateContactManifold(
            shape_a,
            shape_b,
            isom_trans_A_to_W,
            isom_trans_B_to_W,
            temp_con_geom
         );
      }

      void update_aabb_mesh(void)
      {
         geometry::types::aabb_t aabb;
         geometry::types::shape_t shape_a = shape_loaders[0].shape();

         geometry::types::isometricTransform_t trans_B_to_W;
         trans_B_to_W.rotate = shape_loaders[0].trans_B_to_W().rotate;
         trans_B_to_W.translate = shape_loaders[0].trans_B_to_W().translate;

         aabb = geometry::aabbHull(trans_B_to_W, shape_a);

         geometry::types::triangleMesh_t temp_mesh;

         temp_mesh.numVerts = 8;
         temp_mesh.numTriangles = 12;
         for (int i = 0; i < 8; ++i)
         {
            temp_mesh.verts[i][0] = ((i & 1) == 0) ? aabb.vertMin[0] : aabb.vertMax[0];
            temp_mesh.verts[i][1] = ((i & 2) == 0) ? aabb.vertMin[1] : aabb.vertMax[1];
            temp_mesh.verts[i][2] = ((i & 4) == 0) ? aabb.vertMin[2] : aabb.vertMax[2];
         }

         // Obtained by drawing on a cuboid with a Sharpie.
         temp_mesh.triangles[0].vertIds[0] = 0;
         temp_mesh.triangles[0].vertIds[1] = 3;
         temp_mesh.triangles[0].vertIds[2] = 1;
         temp_mesh.triangles[0].normal = Vector3(0.f, 0.f, -1.f);

         temp_mesh.triangles[1].vertIds[0] = 0;
         temp_mesh.triangles[1].vertIds[1] = 2;
         temp_mesh.triangles[1].vertIds[2] = 3;
         temp_mesh.triangles[1].normal = Vector3(0.f, 0.f, -1.f);

         temp_mesh.triangles[2].vertIds[0] = 1;
         temp_mesh.triangles[2].vertIds[1] = 3;
         temp_mesh.triangles[2].vertIds[2] = 7;
         temp_mesh.triangles[2].normal = Vector3(1.f, 0.f, 0.f);

         temp_mesh.triangles[3].vertIds[0] = 1;
         temp_mesh.triangles[3].vertIds[1] = 7;
         temp_mesh.triangles[3].vertIds[2] = 5;
         temp_mesh.triangles[3].normal = Vector3(1.f, 0.f, 0.f);

         temp_mesh.triangles[4].vertIds[0] = 0;
         temp_mesh.triangles[4].vertIds[1] = 1;
         temp_mesh.triangles[4].vertIds[2] = 5;
         temp_mesh.triangles[4].normal = Vector3(0.f, -1.f, 0.f);

         temp_mesh.triangles[5].vertIds[0] = 0;
         temp_mesh.triangles[5].vertIds[1] = 5;
         temp_mesh.triangles[5].vertIds[2] = 4;
         temp_mesh.triangles[5].normal = Vector3(0.f, -1.f, 0.f);

         temp_mesh.triangles[6].vertIds[0] = 2;
         temp_mesh.triangles[6].vertIds[1] = 0;
         temp_mesh.triangles[6].vertIds[2] = 4;
         temp_mesh.triangles[6].normal = Vector3(-1.f, 0.f, 0.f);

         temp_mesh.triangles[7].vertIds[0] = 2;
         temp_mesh.triangles[7].vertIds[1] = 4;
         temp_mesh.triangles[7].vertIds[2] = 6;
         temp_mesh.triangles[7].normal = Vector3(-1.f, 0.f, 0.f);

         temp_mesh.triangles[8].vertIds[0] = 3;
         temp_mesh.triangles[8].vertIds[1] = 2;
         temp_mesh.triangles[8].vertIds[2] = 6;
         temp_mesh.triangles[8].normal = Vector3(0.f, 1.f, 0.f);

         temp_mesh.triangles[9].vertIds[0] = 3;
         temp_mesh.triangles[9].vertIds[1] = 6;
         temp_mesh.triangles[9].vertIds[2] = 7;
         temp_mesh.triangles[9].normal = Vector3(0.f, 1.f, 0.f);

         temp_mesh.triangles[10].vertIds[0] = 6;
         temp_mesh.triangles[10].vertIds[1] = 4;
         temp_mesh.triangles[10].vertIds[2] = 5;
         temp_mesh.triangles[10].normal = Vector3(0.f, 0.f, 1.f);

         temp_mesh.triangles[11].vertIds[0] = 6;
         temp_mesh.triangles[11].vertIds[1] = 5;
         temp_mesh.triangles[11].vertIds[2] = 7;
         temp_mesh.triangles[11].normal = Vector3(0.f, 0.f, 1.f);

         geometry::converters::to_pod(temp_mesh, &shape_a_aabb_mesh_data);
      }
};

#endif
