#include "aabb_tree_ui.hpp"

#include "mesh.hpp"
#include "default_camera_controller.hpp"
#include "geometry_type_converters.hpp"

#include <iostream>

void AabbTreeUi::operator()(void)
{
   no_window();
}

bool AabbTreeUi::no_window(void)
{
   ImGui::Begin("AABB tree tester");

   bool grid_modified = ImGui::Checkbox("Enable grid", &show_grid_);

   if (grid_modified)
   {
      if (show_grid_)
      {
         renderer_.enableGrid();
      }
      else
      {
         renderer_.disableGrid();
      }
   }

   if (ImGui::Button("Generate AABB tree"))
   {
      std::cout << "generating aabb tree\n";
      for (const auto viz_id : tree_viz_ids_)
      {
         renderer_.deleteRenderable(viz_id.id);
      }

      tree_viz_ids_.clear();

      const auto & aabb_entities = allocator_.getQueryEntities(aabb_query_);
      const auto aabbs = allocator_.getComponents<geometry::aabb_t>();

      if (aabb_entities.size() > 0)
      {
         geometry::aabb_meta_t root_meta;
         // root_meta.uid = -1;
         // root_meta.aabb.vertMax.Initialize(1.f, 1.f, 1.f);
         // root_meta.aabb.vertMin.Initialize(-1.f, -1.f, -1.f);
         root_meta.uid = *aabb_entities.begin();
         root_meta.aabb = *aabbs[root_meta.uid];
         tree_.setRoot(root_meta);

         bool first = true;

         for (const auto & aabb_entity : aabb_entities)
         {
            if (first)
            {
               first = false;
               continue;
            }
            geometry::aabb_meta_t new_aabb_meta;
            new_aabb_meta.aabb = *aabbs[aabb_entity];
            new_aabb_meta.uid = aabb_entity;
            geometry::growTree(new_aabb_meta, tree_);
         }

         data_triangleMesh_t viz_mesh = geometry::mesh::loadDefaultShapeMeshData(geometry::types::enumShape_t::CUBE, 1.f);

         for (int i = 0; i < tree_.size(); ++i)
         {
            if (tree_[i].uid < 0)
            {
               aabb_viz_t viz_id;
               viz_id.id = renderer_.addMesh(viz_mesh, {{1.f, 0.f, 0.f, 1.f}}, 1);
               viz_id.enable = true;
               tree_viz_ids_.push_back(viz_id);
               update_viz_mesh(tree_viz_ids_.back().id, tree_[i].aabb);
               renderer_.enableMesh(tree_viz_ids_.back().id);
            }
         }
      }
   }

   if (ImGui::Button("Add AABB"))
   {
      geometry::types::aabb_t default_aabb = {
         {1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f}
      };

      data_triangleMesh_t viz_mesh = geometry::mesh::loadDefaultShapeMeshData(geometry::types::enumShape_t::CUBE, 1.f);

      int aabb_viz_id = renderer_.addMesh(viz_mesh, {{0.f, 1.f, 0.f, 1.f}}, 1);

      auto new_entity = allocator_.addEntity();
      allocator_.addComponent<geometry::types::aabb_t>(
         new_entity, default_aabb
      );
      allocator_.addComponent<aabb_viz_t>(new_entity, {true, aabb_viz_id});

      update_viz_mesh(aabb_viz_id, default_aabb);
   }

   const auto & aabb_entities = allocator_.getQueryEntities(aabb_query_);

   for (const auto aabb_entity : aabb_entities)
   {
      std::string label("aabb ");
      label += std::to_string(aabb_entity);

      if (ImGui::TreeNode(label.c_str()))
      {
         auto & aabb = *allocator_.getComponent<geometry::types::aabb_t>(aabb_entity);

         auto & aabb_viz = *allocator_.getComponent<aabb_viz_t>(aabb_entity);

         Vector3 center = (aabb.vertMax + aabb.vertMin) / 2.f;
         Vector3 max_corner = aabb.vertMax - center;
         Vector3 min_corner = aabb.vertMin - center;

         bool ui_modified = false;

         ui_modified |= ImGui::DragFloat3("center", &(center[0]), 0.01f, -10.f, 10.f);
         ui_modified |= ImGui::DragFloat3("max corner", &(max_corner[0]), 0.01f, 0.f, 10.f);
         ui_modified |= ImGui::DragFloat3("min corner", &(min_corner[0]), 0.01f, -10.f, 0.f);

         aabb.vertMax = center + max_corner;
         aabb.vertMin = center + min_corner;

         if (ui_modified)
         {
            update_viz_mesh(aabb_viz.id, aabb);
         }

         bool enable_modified = ImGui::Checkbox("render", &aabb_viz.enable);

         if (enable_modified)
         {
            if (aabb_viz.enable)
            {
               renderer_.enableMesh(aabb_viz.id);
            }
            else
            {
               renderer_.disableMesh(aabb_viz.id);
            }
         }

         if (ImGui::Button("delete"))
         {
            allocator_.removeEntity(aabb_entity);
            renderer_.deleteRenderable(aabb_viz.id);
         }

         ImGui::TreePop();
      }
   }

   ImGui::End();

   ImGui::Begin("Tree visualizer");

   // for (auto aabb_viz_id : tree_viz_ids_)
   for (int i = 0; i < tree_viz_ids_.size(); ++i)
   {
      auto & aabb_viz_id = tree_viz_ids_[i];
      std::string viz_id_label("tree node id ");
      viz_id_label += std::to_string(i);
      if (ImGui::TreeNode(viz_id_label.c_str()))
      {
         // renderer_.

         bool enable_modified = ImGui::Checkbox("render", &aabb_viz_id.enable);

         if (enable_modified)
         {
            if (aabb_viz_id.enable)
            {
               renderer_.enableMesh(aabb_viz_id.id);
            }
            else
            {
               renderer_.disableMesh(aabb_viz_id.id);
            }
         }

         ImGui::TreePop();
      }
   }

   ImGui::End();

   return false;
}

void AabbTreeUi::update_viz_mesh(
   const int viz_id, const geometry::types::aabb_t & aabb
)
{
   geometry::types::triangleMesh_t temp_mesh;
   temp_mesh.numVerts = 8;
   temp_mesh.numTriangles = 12;
   for (int i = 0; i < 8; ++i)
   {
      temp_mesh.verts[i][0] = ((i & 1) == 0) ? aabb.vertMin[0] : aabb.vertMax[0];
      temp_mesh.verts[i][1] = ((i & 2) == 0) ? aabb.vertMin[1] : aabb.vertMax[1];
      temp_mesh.verts[i][2] = ((i & 4) == 0) ? aabb.vertMin[2] : aabb.vertMax[2];
   }

   // Obtained by drawing on a rectangle with a Sharpie.
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

   data_triangleMesh_t viz_mesh;
   geometry::converters::to_pod(temp_mesh, &viz_mesh);

   renderer_.updateMesh(viz_id, viz_mesh);
}

int main(void)
{
   viz::VizRenderer renderer;
   AabbTreeUi gui("aabb tree", renderer);
   viz::GuiCallbackBase * gui_ref = &gui;

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   controller.setCameraSpeed(0.03f);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   while (true)
   {
      if (renderer.draw(camera, gui_ref))
      {
         break;
      }
   }

   return 0;
}
