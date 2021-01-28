#include "ld52_gui.hpp"

void LD52Gui::operator()(void)
{
   const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
   ImGui::SetNextWindowPos(ImVec2(main_viewport->WorkPos.x + 150, main_viewport->WorkPos.y + 20), ImGuiCond_FirstUseEver);
   ImGui::SetNextWindowSize(ImVec2(550, 400), ImGuiCond_FirstUseEver);

   ImGui::SetNextWindowSizeConstraints(ImVec2(250, 100), ImVec2(550, 400));

   ImGui::Begin("Jumping bean harvester GUI", nullptr, ImGuiWindowFlags_NoCollapse);

   if (state_.show_instructions)
   {
      instructions();
   }
   else if (state_.lives_remaining > 0 && !state_.win)
   {
      no_window();
   }
   else if (state_.lives_remaining == 0)
   {
      game_over();
   }
   else if (state_.win)
   {
      win();
   }

   ImGui::End();
}

bool LD52Gui::no_window(void)
{
   ImGui::Text(
      "Score: %f", state_.score
   );

   ImGui::Text(
      "Lives remaining: %u", state_.lives_remaining
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

      ImGui::Text("Right mouse click: Grab grabbable bean");
   }

   return false;
}

void LD52Gui::instructions(void)
{
   ImGui::TextColored(ImVec4(0.8f, 0.1f, 0.9f, 1.f), "Instructions");
   ImGui::Separator();
   ImGui::Text("You are a bean harvester.");
   ImGui::Text("Drive your combine, grab beans that turn");
   ImGui::SameLine();
   ImGui::TextColored(ImVec4(0.f, 0.647f, 1.f, 1.f), "blue");
   ImGui::Text("When you've grabbed a bean, it will turn");
   ImGui::SameLine();
   ImGui::TextColored(ImVec4(1.f, 0.147f, 0.2f, 1.f), "red");
   ImGui::Text("Drop all of the beans into the bucket at the top of the map to win.");
   ImGui::Text("Jumping beans are worth 50 points and are");
   ImGui::SameLine();
   ImGui::TextColored(ImVec4(0.28235f, 0.047058f, 0.41176f, 1.f), "aubergine");
   ImGui::Text("Stationary beans are worth 5 points.");
   ImGui::TextWrapped("You have five lives. Falling off the map or resetting the vehicle costs one life.");

   ImGui::TextColored(ImVec4(0.8f, 0.1f, 0.9f, 1.f), "Controls");
   ImGui::Separator();
   ImGui::Text("Exit          -    Esc");
   ImGui::Text("Forward       -    W");
   ImGui::Text("Turn left     -    A");
   ImGui::Text("Backward      -    S");
   ImGui::Text("Turn right    -    D");
   ImGui::Text("Reset vehicle - M");
   ImGui::Text("Rotate camera - Left mouse click and drag");
   ImGui::Text("Grab grabbable bean - Left shift");
   ImGui::Text("Grab grabbable bean - Right mouse click and hold");

   if (ImGui::Button("Continue to game"))
   {
      state_.show_instructions = false;
   }
}

void LD52Gui::game_over(void)
{
   ImGui::TextColored(ImVec4(0.8f, 0.1f, 0.1f, 1.f), "GAME OVER");
   ImGui::Text("Press Escape to exit");
}

void LD52Gui::win(void)
{
   ImGui::TextColored(ImVec4(0.1f, 0.8f, 0.1f, 1.f), "YOU WIN");
   ImGui::Text("Press Escape to exit");
}
