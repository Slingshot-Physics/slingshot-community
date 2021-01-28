#include "camera_control_gui.hpp"

bool CameraGui::no_window(void)
{
   bool ui_modified = false;
   ui_modified |= ImGui::Checkbox("Show grid lines", &state_.show_grid);

   ImGui::Text(
      "Camera direction: %0.4f, %0.4f %0.4f",
      state_.camera_direction[0],
      state_.camera_direction[1],
      state_.camera_direction[2]
   );

   ImGui::Text("Camera speed:");
   ImGui::SameLine();
   ui_modified |= ImGui::SliderFloat("Camspeed", &state_.camera_speed, 0.5f, 15.f);

   ImGui::Text("Num meshes: %d", state_.num_meshes);

   ImGui::Text("Num draw calls: %d", state_.num_draw_calls);

   ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

   return ui_modified;
}

