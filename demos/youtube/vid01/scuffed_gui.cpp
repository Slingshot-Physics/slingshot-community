#include "scuffed_gui.hpp"

void ScuffedGui::operator()(void)
{
   const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
   ImGui::SetNextWindowPos(ImVec2(main_viewport->WorkPos.x + 150, main_viewport->WorkPos.y + 20), ImGuiCond_FirstUseEver);
   ImGui::SetNextWindowSize(ImVec2(550, 400), ImGuiCond_FirstUseEver);

   ImGui::SetNextWindowSizeConstraints(ImVec2(250, 100), ImVec2(550, 400));

   ImGui::Begin("Scuffed-It League GUI", nullptr, ImGuiWindowFlags_NoCollapse);

   no_window();

   ImGui::End();
}

bool ScuffedGui::no_window(void)
{
   ImGui::Text(
      "Score: %f", state_.score
   );

   if (ImGui::CollapsingHeader("Controls"))
   {
      ImGui::Text("Esc - Exit");

      ImGui::Text("W - Forward");
      ImGui::Text("A - Turn left");
      ImGui::Text("S - Backward");
      ImGui::Text("D - Turn right");

      ImGui::Text("M - Reset vehicle");

      ImGui::Text("Left mouse click and drag: Rotate camera");
   }

   return false;
}
