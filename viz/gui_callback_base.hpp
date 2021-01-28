#ifndef GUICALLBACKBASEHEADER
#define GUICALLBACKBASEHEADER

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

namespace viz
{

class GuiCallbackBase
{

   public:
      // This should contain the ImGui Begin/End calls, sandwiching the
      // `no_window` method.
      virtual void operator()(void) = 0;

      // This should contain the ImGui widget calls. Returns true if the
      // widgets are interacted with, false otherwise.
      virtual bool no_window(void) = 0;

};

}

#endif
