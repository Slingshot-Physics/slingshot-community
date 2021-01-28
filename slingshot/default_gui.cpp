#include "default_gui.hpp"

#include "glad/glad.h"

#include <cmath>

void DefaultGuiCallback::operator()(void)
{
   ImGui::Begin("Default GUI Controls");

   no_window();

   ImGui::End();

   glClearColor(
      clear_color_.x * clear_color_.w,
      clear_color_.y * clear_color_.w,
      clear_color_.z * clear_color_.w,
      clear_color_.w
   );
}

bool DefaultGuiCallback::no_window(void)
{
   bool ui_modified = false;
   ui_modified |= ImGui::Checkbox("Play", &state_.run_sim);

   ImGui::ColorEdit3("clear color", (float*)&clear_color_);

   state_.step_one = false;
   if (ImGui::Button("Step one frame"))
   {
      state_.run_sim = false;
      state_.step_one = true;
      ui_modified |= true;
   }

   ImGui::Checkbox("Show grid lines", &state_.show_grid);

   ImGui::Text(
      "Camera direction: %0.4f, %0.4f %0.4f",
      state_.camera_direction[0],
      state_.camera_direction[1],
      state_.camera_direction[2]
   );

   ImGui::Text("Camera speed:");
   ImGui::SameLine();
   ui_modified |= ImGui::SliderFloat("Camspeed", &state_.camera_speed, 0.05f, 2.f);

   float theta = atan2f(
      state_.light_direction[1],
      state_.light_direction[0]
   );

   float psi = atan2f(
      state_.light_direction[2],
      sqrtf(
         state_.light_direction[0] * state_.light_direction[0] + \
         state_.light_direction[1] * state_.light_direction[1]
      )
   );

   bool theta_changed = ImGui::DragFloat("Directional light azimuth", &theta, 0.0125f);
   bool psi_changed = ImGui::DragFloat("Directional light elevation", &psi, 0.0125f, 0.f, M_PI/2.f);

   ui_modified |= (theta_changed || psi_changed);

   if (theta_changed || psi_changed)
   {
      state_.light_direction = Vector3(
         cosf(theta) * cosf(psi), sinf(theta) * cosf(psi), sinf(psi)
      );
   }

   ImGui::Text("Num meshes: %d", state_.num_meshes);

   ImGui::Text("Num draw calls: %d", state_.num_draw_calls);

   ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

   return ui_modified;
}
