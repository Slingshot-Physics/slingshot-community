#ifndef MESH_FEATURE_UI_HEADER
#define MESH_FEATURE_UI_HEADER

#include "gui_callback_base.hpp"

#include "geometry_type_converters.hpp"

#include "polyhedron_loader_ui.hpp"
#include "transform_ui.hpp"

class MeshFeature : public viz::GuiCallbackBase
{
   public:

      MeshFeature(void)
         : poly_loader("polyhedron", {0.f, 0.f, 0.f})
         , transform_ui("plane transform")
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

      PolyhedronLoader poly_loader;

      TransformUi transform_ui;

      data_triangleMesh_t polyhedron_render_data;

      data_triangleMesh_t plane_render_data;

      geometry::types::transform_t trans_P_to_W;

      geometry::types::polygon50_t best_feature;

   private:

      geometry::types::plane_t query_plane_;

      geometry::types::gaussMapMesh_t gauss_map_;

      void update_render_meshes(void)
      {
         geometry::types::triangleMesh_t temp_mesh = poly_loader.mesh();
         geometry::converters::to_pod(temp_mesh, &polyhedron_render_data);
      }

};

#endif
