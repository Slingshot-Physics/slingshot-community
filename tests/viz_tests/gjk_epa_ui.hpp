#include "gui_callback_base.hpp"
#include "dynamic_array.hpp"
#include "vector3.hpp"

#include "polyhedron_loader_ui.hpp"

#include <algorithm>
#include <vector>

class MeddlingGui : public viz::GuiCallbackBase
{
   public:
      MeddlingGui(void)
         : ui_modified_(false)
         , file_path_("")
         , button_color_(0.f, 1.f, 0.f, 1.f)
         , run_gjk(true)
         , show_gjk_points(true)
         , run_epa(false)
         , show_epa_points(false)
         , show_collision_normal(false)
         , run_manifold(false)
         , show_manifold(false)
         , poly_loaders{
            {"polyhedron A", Vector3(0.f, 0.f, 2.f)},
            {"polyhedron B", Vector3(0.f, 0.f, -2.f)}
         }
         , contacts(32)
      {
         epa_mesh_data.numTriangles = 0;
         epa_mesh_data.numVerts = 0;

         for (int i = 0; i < 2; ++i)
         {
            update_render_mesh(i);
         }

         run_gjk_alg();
      }

      void operator()(void);

      bool no_window(void);

      bool ui_modified(void)
      {
         return ui_modified_;
      }

   private:

      bool ui_modified_;

      char file_path_[256];

      ImVec4 button_color_;

      // Load a config file.
      bool load_file(void);

      // Executes the GJK algorithm, calculates the closest points between the
      // two bodies, keeps the result.
      void run_gjk_alg(void);

      // Executes EPA, calculates the contact points, keeps the result.
      // "run expanding polytope algorithm algorithm" :|
      void run_epa_alg(void);

      void run_contact_manifold_alg(void);

      void update_render_mesh(int index)
      {
         geometry::types::triangleMesh_t temp_mesh = poly_loaders[index].mesh();
         geometry::converters::to_pod(temp_mesh, &render_meshes_data[index]);
      }

   public:

      bool run_gjk;

      bool show_gjk_points;

      bool run_epa;

      bool show_epa_points;

      bool show_collision_normal;

      bool run_manifold;

      bool show_manifold;

      Vector3 closest_pt_a_world;

      Vector3 closest_pt_b_world;

      geometry::types::gjkResult_t gjk_out;

      Vector3 contact_pt_a_world;

      Vector3 contact_pt_b_world;

      Vector3 collision_normal_world;

      geometry::types::epaResult_t epa_out;

      geometry::types::epa::epaMesh_t mesh_md_W;

      PolyhedronLoader poly_loaders[2];

      data_triangleMesh_t render_meshes_data[2];

      data_triangleMesh_t epa_mesh_data;

      DynamicArray<Vector3> contacts;

};
