#include "callbacks.hpp"

#include "hid_interface.hpp"

namespace viz
{

namespace callbacks
{

   void frameBufferSize(GLFWwindow * window, int width, int height)
   {
      int frameBufferWidth, frameBufferHeight;
      glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
      glViewport(0, 0, frameBufferWidth, frameBufferHeight);

      void * userPointer = glfwGetWindowUserPointer(window);
      if (userPointer != nullptr)
      {
         HIDInterface * hid = static_cast<HIDInterface *>(userPointer);
         hid->frameBufferSize_cb(width, height);
      }
   }

   void windowSize(GLFWwindow * window, int width, int height)
   {
      void * userPointer = glfwGetWindowUserPointer(window);
      if (userPointer != nullptr)
      {
         HIDInterface * hid = static_cast<HIDInterface *>(userPointer);
         hid->windowSize_cb(width, height);
      }
   }

   void keyboard(
      GLFWwindow * window, int key, int scancode, int action, int mods
   )
   {

   }

   void mouseButton(GLFWwindow * window, int button, int action, int mods)
   {
      void * userPointer = glfwGetWindowUserPointer(window);
      if (userPointer != nullptr)
      {
         HIDInterface * hid = static_cast<HIDInterface *>(userPointer);
         hid->mouseButton_cb(button, action);
      }
   }

   void mouseMove(GLFWwindow * window, double xpos, double ypos)
   {
      void * userPointer = glfwGetWindowUserPointer(window);
      if (userPointer != nullptr)
      {
         HIDInterface * hid = static_cast<HIDInterface *>(userPointer);
         hid->mouseMove_cb(xpos, ypos);
      }
   }

}

}
