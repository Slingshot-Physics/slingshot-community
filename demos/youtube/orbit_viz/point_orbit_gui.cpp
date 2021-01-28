#include "point_orbit_gui.hpp"

void PointOrbitGui::operator()(void)
{
   ImGui::Begin("Point orbit GUI", nullptr);

   no_window();

   ImGui::End();
}

bool PointOrbitGui::no_window(void)
{
   bool ui_modified = false;

   ui_modified |= ImGui::Checkbox("Play", &state_.run_sim);

   ui_modified |= ImGui::Checkbox("Show grid", &state_.show_grid);

   ui_modified |= ImGui::DragFloat("radius", &state_.orbit_radius, 0.01f, 0.f, 20.f);

   ui_modified |= ImGui::DragFloat("frequency", &state_.orbit_frequency, 0.01f, -100.f, 100.f);

   ui_modified |= ImGui::DragFloat("height", &state_.orbit_height, 0.1f);

   ui_modified |= ImGui::DragFloat3("orbit point", &(state_.look_point[0]), 0.025f);

   bool theta_changed = ImGui::DragFloat("Directional light azimuth", &light_theta_deg_, 0.1f);
   bool psi_changed = ImGui::DragFloat("Directional light elevation", &light_psi_deg_, 0.1f, 15.f, 90.f);

   ui_modified |= (theta_changed || psi_changed);

   float theta_rad = light_theta_deg_ * deg2rad;

   float psi_rad = light_psi_deg_ * deg2rad;

   if (theta_changed || psi_changed)
   {
      state_.light_direction = Vector3(
         cosf(theta_rad) * cosf(psi_rad),
         sinf(theta_rad) * cosf(psi_rad),
         sinf(psi_rad)
      );
   }

   return ui_modified;
}
