#ifndef SHAPE_SHAPE_COLLISION_UI
#define SHAPE_SHAPE_COLLISION_UI

#include "gui_callback_base.hpp"

#include "aabb_hull.hpp"
#include "epa.hpp"
#include "gjk.hpp"

#include "collision_contacts.hpp"
#include "dynamic_array.hpp"
#include "epa_types.hpp"
#include "slingshot_types.hpp"
#include "geometry_type_converters.hpp"

#include "shape_loader_ui.hpp"

#include <string>

void sphere_sphere(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

void sphere_capsule(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

void sphere_cylinder(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

void capsule_sphere(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

void capsule_capsule(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

void capsule_cylinder(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

void cylinder_sphere(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

void cylinder_capsule(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

void cylinder_cylinder(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::shape_t & shape_a,
   const geometry::types::shape_t & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
);

Vector3 rescaleEpaNormal(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const geometry::types::epaResult_t & epa_out
)
{
   Vector3 poly_a_pos = trans_A_to_W.translate;
   Vector3 poly_b_pos = trans_B_to_W.translate;

   Vector3 md_cm_vec_W = poly_b_pos - poly_a_pos;
   float sign = -1.f + (md_cm_vec_W.dot(epa_out.p) > 0.f) * 2.f;

   Vector3 collision_W = (sign * epa_out.p).unitVector();
   return collision_W;
}

template <typename ShapeA_T, typename ShapeB_T>
void gjk_epa_manifold(
   const geometry::types::transform_t & trans_A_to_W,
   const geometry::types::transform_t & trans_B_to_W,
   const ShapeA_T & shape_a,
   const ShapeB_T & shape_b,
   geometry::types::gjkResult_t & gjk_out,
   geometry::types::epaResult_t & epa_out,
   DynamicArray<Vector3> & contacts
)
{
   contacts.clear();

   gjk_out = geometry::gjk::alg(
      trans_A_to_W, trans_B_to_W, shape_a, shape_b
   );

   if (gjk_out.intersection)
   {
      geometry::types::minkowskiDiffSimplex_t tetra_simplex = gjk_out.minSimplex;

      geometry::epa::expandGjkSimplex(
         trans_A_to_W, trans_B_to_W, shape_a, shape_b, tetra_simplex
      );

      epa_out = geometry::epa::smoothAlg(
         trans_A_to_W,
         trans_B_to_W,
         tetra_simplex,
         shape_a,
         shape_b,
         1.f - 5e-5f
      );

      // std::cout << "gjk collision\n";
      // std::cout << "epa collided? " << epa_out.collided << "\n";
   }
   else
   {
      // std::cout << "gjk no collision\n";
      epa_out.collided = false;
      return;
   }

   if (!epa_out.collided)
   {
      return;
   }

   Vector3 collision_W = rescaleEpaNormal(
      trans_A_to_W,
      trans_B_to_W,
      epa_out
   );

   oy::types::contactGeometry_t contact_geom;
   contact_geom.initialize(epa_out);

   oy::types::collisionContactManifold_t mani = oy::collision::calculateContactManifold(
      shape_a,
      shape_b,
      trans_A_to_W,
      trans_B_to_W,
      contact_geom
   );

   for (unsigned int i = 0; i < mani.numContacts; ++i)
   {
      contacts.push_back(mani.bodyAContacts[i]);
   }
}

class ShapeShapeCollision : public viz::GuiCallbackBase
{
   public:
      ShapeShapeCollision(void)
         : shape_loaders{
            {"shape A", {0.f, 0.f, 2.f}},
            {"shape B", {0.f, 0.f, -2.f}}
         }
         , contacts_W(64)
         , ui_modified_(true)
         , button_color_(0.f, 1.f, 0.f, 1.f)
      {
         update_render_meshes();

         for (int i = 0; i < 10; ++i)
         {
            for (int j = 0; j < 10; ++j)
            {
               call_matrix_[i][j] = nullptr;
            }
         }

         int i, j;

         i = static_cast<int>(geometry::types::enumShape_t::SPHERE);
         j = static_cast<int>(geometry::types::enumShape_t::SPHERE);
         call_matrix_[i][j] = capsule_capsule;

         i = static_cast<int>(geometry::types::enumShape_t::SPHERE);
         j = static_cast<int>(geometry::types::enumShape_t::CAPSULE);
         call_matrix_[i][j] = sphere_capsule;

         i = static_cast<int>(geometry::types::enumShape_t::SPHERE);
         j = static_cast<int>(geometry::types::enumShape_t::CYLINDER);
         call_matrix_[i][j] = sphere_cylinder;

         i = static_cast<int>(geometry::types::enumShape_t::CAPSULE);
         j = static_cast<int>(geometry::types::enumShape_t::SPHERE);
         call_matrix_[i][j] = capsule_sphere;

         i = static_cast<int>(geometry::types::enumShape_t::CAPSULE);
         j = static_cast<int>(geometry::types::enumShape_t::CAPSULE);
         call_matrix_[i][j] = capsule_capsule;

         i = static_cast<int>(geometry::types::enumShape_t::CAPSULE);
         j = static_cast<int>(geometry::types::enumShape_t::CYLINDER);
         call_matrix_[i][j] = capsule_cylinder;

         i = static_cast<int>(geometry::types::enumShape_t::CYLINDER);
         j = static_cast<int>(geometry::types::enumShape_t::SPHERE);
         call_matrix_[i][j] = cylinder_sphere;

         i = static_cast<int>(geometry::types::enumShape_t::CYLINDER);
         j = static_cast<int>(geometry::types::enumShape_t::CAPSULE);
         call_matrix_[i][j] = cylinder_capsule;

         i = static_cast<int>(geometry::types::enumShape_t::CYLINDER);
         j = static_cast<int>(geometry::types::enumShape_t::CYLINDER);
         call_matrix_[i][j] = cylinder_cylinder;

         collision_multiplexor();
      }

      void operator()(void)
      {
         ImGui::Begin("shape-shape collision");
         no_window();
         ImGui::End();
      }

      bool no_window(void);

      bool ui_modified(void) const
      {
         return ui_modified_;
      }

      const geometry::types::gjkResult_t & gjk_out(void) const
      {
         return gjk_out_;
      }

      const geometry::types::epaResult_t & epa_out(void) const
      {
         return epa_out_;
      }

      ShapeLoader shape_loaders[2];

      data_triangleMesh_t shape_render_data[2];

      data_triangleMesh_t shape_a_aabb_mesh_data;

      Vector3 gjk_points_W[2];

      Vector3 epa_contact_points_W[2];

      Vector3 epa_collision_normal_W;

      DynamicArray<Vector3> contacts_W;

   private:
      typedef void (*ShapeShapeCollision_f)(
         const geometry::types::transform_t & trans_A_to_W,
         const geometry::types::transform_t & trans_B_to_W,
         const geometry::types::shape_t & shape_a,
         const geometry::types::shape_t & shape_b,
         geometry::types::gjkResult_t & gjk_out,
         geometry::types::epaResult_t & epa_out,
         DynamicArray<Vector3> & contacts
      );

      bool ui_modified_;

      std::string file_path_;

      ImVec4 button_color_;

      geometry::types::gjkResult_t gjk_out_;

      geometry::types::epaResult_t epa_out_;

      ShapeShapeCollision_f call_matrix_[10][10];

      void update_render_meshes(void)
      {
         geometry::types::triangleMesh_t temp_mesh;
         for (int i = 0; i < 2; ++i)
         {
            temp_mesh = shape_loaders[i].mesh();
            geometry::converters::to_pod(temp_mesh, &(shape_render_data[i]));
         }

         geometry::types::aabb_t aabb;
         geometry::types::shape_t shape_a = shape_loaders[0].shape();

         aabb = geometry::aabbHull(
            shape_loaders[0].trans_B_to_W(), shape_a
         );

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

      void collision_multiplexor(void);

      bool load_file(void);
};

#endif
