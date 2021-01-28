#ifndef CALLBACKS_HEADER
#define CALLBACKS_HEADER

#include "gl_common.hpp"

namespace viz
{

namespace callbacks
{

   void frameBufferSize(GLFWwindow * window, int width, int height);

   void windowSize(GLFWwindow * window, int width, int height);

   void keyboard(
      GLFWwindow * window, int key, int scancode, int action, int mods
   );

   void mouseButton(GLFWwindow * window, int button, int action, int mods);

   void mouseMove(GLFWwindow * window, double xpos, double ypos);

}

}

#endif
