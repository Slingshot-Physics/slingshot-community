#ifndef SHAPE_FEATURE_UI_HEADER
#define SHAPE_FEATURE_UI_HEADER

#include "gui_callback_base.hpp"

#include "geometry_type_converters.hpp"

#include "shape_loader_ui.hpp"
#include "transform_ui.hpp"

class MeshFeature : public viz::GuiCallbackBase
{
   public:

      MeshFeature(void)
         : shape_loader("shape", {0.f, 0.f, 0.f})
         , transform_ui("plane transform")
         , ui_modified_(false)
         , query_plane_{
            {0.f, 0.f, 2.f}, {0.f, 0.f, 1.f}
         }
      {
         update_render_meshes();

         plane_render_data.numVerts = 4;
         plane_render_data.numTriangles = 2;

         plane_render_data.verts[0].v[0] = -3.f;
         plane_render_data.verts[0].v[1] = -3.f;
         plane_render_data.verts[0].v[2] =  0.f;

         plane_render_data.verts[1].v[0] = -3.f;
         plane_render_data.verts[1].v[1] =  3.f;
         plane_render_data.verts[1].v[2] =  0.f;

         plane_render_data.verts[2].v[0] =  3.f;
         plane_render_data.verts[2].v[1] =  3.f;
         plane_render_data.verts[2].v[2] =  0.f;

         plane_render_data.verts[3].v[0] =  3.f;
         plane_render_data.verts[3].v[1] = -3.f;
         plane_render_data.verts[3].v[2] =  0.f;

         plane_render_data.triangles[0].normal.v[0] = 0.f;
         plane_render_data.triangles[0].normal.v[1] = 0.f;
         plane_render_data.triangles[0].normal.v[2] = 1.f;

         plane_render_data.triangles[1].normal.v[0] = 0.f;
         plane_render_data.triangles[1].normal.v[1] = 0.f;
         plane_render_data.triangles[1].normal.v[2] = 1.f;

         plane_render_data.triangles[0].vertIds[0] = 0;
         plane_render_data.triangles[0].vertIds[1] = 1;
         plane_render_data.triangles[0].vertIds[2] = 2;

         plane_render_data.triangles[1].vertIds[0] = 0;
         plane_render_data.triangles[1].vertIds[1] = 2;
         plane_render_data.triangles[1].vertIds[2] = 3;

         best_feature.numVerts = 0;
      }

      void operator()(void)
      {
         ImGui::Begin("shape-convex polyhedron collision");
         no_window();
         ImGui::End();
      }

      bool no_window(void);

      const geometry::types::plane_t & query_plane(void) const
      {
         return query_plane_;
      }

      bool ui_modified(void)
      {
         return ui_modified_;
      }

      ShapeLoader shape_loader;

      TransformUi transform_ui;

      data_triangleMesh_t shape_render_data;

      data_triangleMesh_t plane_render_data;

      geometry::types::transform_t trans_P_to_W;

      geometry::types::polygon50_t best_feature;

   private:
      bool ui_modified_;

      geometry::types::plane_t query_plane_;

      geometry::types::gaussMapMesh_t gauss_map_;

      void update_render_meshes(void)
      {
         geometry::types::triangleMesh_t temp_mesh = shape_loader.mesh();
         geometry::converters::to_pod(temp_mesh, &shape_render_data);
      }

};

#endif
