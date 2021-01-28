#ifndef SHAPE_POLYHEDRON_COLLISION_UI_HEADER
#define SHAPE_POLYHEDRON_COLLISION_UI_HEADER

#include "gui_callback_base.hpp"

#include "dynamic_array.hpp"
#include "epa_types.hpp"
#include "slingshot_types.hpp"
#include "geometry_type_converters.hpp"

#include "polyhedron_loader_ui.hpp"
#include "shape_loader_ui.hpp"

class ImplicitShapeCollision : public viz::GuiCallbackBase
{
   public:
      ImplicitShapeCollision(void)
         : ui_modified_(false)
         , show_gjk_(true)
         , show_epa_(true)
         , show_contacts_(false)
         , shape_loader("implicit shape", {0.f, 0.f, 2.f})
         , poly_loader("convex polyhedron", {0.f, 0.f, -2.f})
         , contacts_W(64)
      {
         update_render_meshes();
      }

      void operator()(void)
      {
         ImGui::Begin("shape-convex polyhedron collision");
         no_window();
         ImGui::End();
      }

      bool no_window(void);

      void show_gjk(bool enable)
      {
         show_gjk_ = enable;
      }

      void show_epa(bool enable)
      {
         show_epa_ = enable;
      }

      const bool show_gjk(void)
      {
         return show_gjk_;
      }

      const bool show_epa(void)
      {
         return show_epa_;
      }

      const bool show_contacts(void)
      {
         return show_contacts_;
      }

      const geometry::types::gjkResult_t & gjk_out(void) const
      {
         return gjk_out_;
      }

      const geometry::types::epaResult_t & epa_out(void) const
      {
         return epa_out_;
      }

      const bool ui_modified(void) const
      {
         return ui_modified_;
      }

      data_triangleMesh_t shape_render_data;

      data_triangleMesh_t polyhedron_render_data;

      ShapeLoader shape_loader;

      PolyhedronLoader poly_loader;

      Vector3 gjk_point_a_W;

      Vector3 gjk_point_b_W;

      Vector3 epa_contact_point_a_W;

      Vector3 epa_contact_point_b_W;

      Vector3 epa_collision_normal_W;

      DynamicArray<Vector3> contacts_W;

   private:
      bool ui_modified_;

      bool show_gjk_;

      bool show_epa_;

      bool show_contacts_;

      geometry::types::gjkResult_t gjk_out_;

      geometry::types::epaResult_t epa_out_;

      Vector3 rescaleEpaNormal(void)
      {
         Vector3 poly_a_pos = shape_loader.trans_B_to_W().translate;
         Vector3 poly_b_pos = poly_loader.trans_B_to_W().translate;

         Vector3 md_cm_vec_W = poly_b_pos - poly_a_pos;
         float sign = -1.f + (md_cm_vec_W.dot(epa_out_.p) > 0.f) * 2.f;

         Vector3 collision_W = (sign * epa_out_.p).unitVector();
         return collision_W;
      }

      void update_render_meshes(void)
      {
         geometry::types::triangleMesh_t temp_mesh = shape_loader.mesh();
         geometry::converters::to_pod(temp_mesh, &shape_render_data);
         temp_mesh = poly_loader.mesh();
         geometry::converters::to_pod(temp_mesh, &polyhedron_render_data);
      }
};

#endif
