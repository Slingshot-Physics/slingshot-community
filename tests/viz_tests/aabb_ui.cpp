#include "aabb_ui.hpp"

#include "geometry_type_converters.hpp"

#include <algorithm>

void AABBUI::operator()(void)
{
   ImGui::Begin("AABB UI");

   no_window();

   ImGui::End();
}

bool AABBUI::no_window(void)
{
   bool ui_modified = false;
   char corner_1_label[100];
   sprintf(corner_1_label, "%s corner 1", prefix_.c_str());
   ui_modified |= ImGui::DragFloat3(corner_1_label, &(corners_[0][0]), 0.01f, -10.f, 10.f);

   char corner_2_label[100];
   sprintf(corner_2_label, "%s corner 2", prefix_.c_str());
   ui_modified |= ImGui::DragFloat3(corner_2_label, &(corners_[1][0]), 0.01f, -10.f, 10.f);

   char center_label[100];
   sprintf(center_label, "%s center", prefix_.c_str());
   ui_modified |= ImGui::DragFloat3(center_label, &center_[0], 0.1f, -10.f, 10.f);

   if (ui_modified)
   {
      aabb.vertMax = Vector3(
         std::max(corners_[0][0], corners_[1][0]),
         std::max(corners_[0][1], corners_[1][1]),
         std::max(corners_[0][2], corners_[1][2])
      ) + center_;

      aabb.vertMin = Vector3(
         std::min(corners_[0][0], corners_[1][0]),
         std::min(corners_[0][1], corners_[1][1]),
         std::min(corners_[0][2], corners_[1][2])
      ) + center_;

      update_viz_mesh();
   }

   return ui_modified;
}

void AABBUI::update_viz_mesh(void)
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

   geometry::converters::to_pod(temp_mesh, &viz_mesh);
}
