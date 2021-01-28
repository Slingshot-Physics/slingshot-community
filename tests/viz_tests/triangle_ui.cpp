#include "triangle_ui.hpp"

#include "attitudeutils.hpp"
#include "geometry_type_converters.hpp"
#include "mesh_ops.hpp"

#include <algorithm>

void TriangleUI::operator()(void)
{
   ImGui::Begin("Triangle UI");

   no_window();

   ImGui::End();
}

bool TriangleUI::no_window(void)
{
   bool ui_modified = false;
   for (int i = 0; i < num_points_; ++i)
   {
      char point_label[100];
      sprintf(point_label, "%s %i", prefix_.c_str(), i);
      ui_modified |= ImGui::DragFloat3(point_label, &(raw_triangle_.verts[i][0]), 0.01f, -10.f, 10.f, "%0.7f");
   }

   ImGui::Separator();

   char center_label[100];
   sprintf(center_label, "%s center", prefix_.c_str());
   ui_modified |= ImGui::DragFloat3(
      center_label, &center[0], 0.01f, -10.f, 10.f
   );

   char rpy_label[100];
   sprintf(rpy_label, "%s rpy", prefix_.c_str());
   ui_modified |= ImGui::DragFloat3(
      rpy_label, &(rpy[0]), 0.025f, -1.f * M_PI, M_PI
   );

   rpy[1] = std::max(
      std::min(rpy[1], (float )M_PI/2.f),
      -1.f * (float )M_PI/2.f
   );

   if (ui_modified)
   {
      update_transformed_points();
   }

   return ui_modified;
}

void TriangleUI::update_transformed_points(void)
{
   Matrix33 rot_mat = frd2NedMatrix(rpy);
   for (int i = 0; i < num_points_; ++i)
   {
      triangle.verts[i] = rot_mat * raw_triangle_.verts[i] + center;
   }

   viz_mesh.numTriangles = 1;
   viz_mesh.numVerts = 3;
   viz_mesh.triangles[0].vertIds[0] = 0;
   viz_mesh.triangles[0].vertIds[1] = 1;
   viz_mesh.triangles[0].vertIds[2] = 2;
   geometry::converters::to_pod(triangle.verts[0], &(viz_mesh.verts[0]));
   geometry::converters::to_pod(triangle.verts[1], &(viz_mesh.verts[1]));
   geometry::converters::to_pod(triangle.verts[2], &(viz_mesh.verts[2]));

   // Normal isn't so important in this case, but it's added for completeness.
   Vector3 normal = rot_mat * Vector3(0.f, 0.f, 1.f);

   geometry::converters::to_pod(normal, &(viz_mesh.triangles[0].normal));
}
