#include "editor.hpp"

#include <chrono>
#include <thread>

void Editor::loop(void)
{
   camera_controller_.cameraSpeed() = 0.25f;
   while(true)
   {
      viz::GuiCallbackBase * gui_base_ = &gui_;
      stop_sim_ |= renderer_.draw(camera_, gui_base_);
      if (stop_sim_)
      {
         break;
      }

      // Normally I'd say I'm better than this, but I'm really really not.
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
   }
}
