#ifndef HID_INTERFACE_HEADER
#define HID_INTERFACE_HEADER

#include "data_viz_config.h"
#include "gl_common.hpp"

namespace viz
{
   class HIDInterface
   {
      public:
         virtual void initialize(const data_vizConfig_t & viz_config_data) = 0;

         virtual void mouseButton_cb(int button, int action) = 0;

         virtual void mouseMove_cb(double xpos, double ypos) = 0;

         virtual void keyboard_cb(GLFWwindow * window) = 0;

         virtual void frameBufferSize_cb(int width, int height) = 0;

         virtual void windowSize_cb(int width, int height) = 0;
   };
}

#endif
