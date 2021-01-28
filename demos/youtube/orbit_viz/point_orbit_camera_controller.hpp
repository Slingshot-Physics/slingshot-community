#ifndef POINT_ORBIT_CAMERA_CONTROLLER_HEADER
#define POINT_ORBIT_CAMERA_CONTROLLER_HEADER

#include "gl_common.hpp"

#include "camera.hpp"
#include "hid_interface.hpp"
#include "imgui.h"

namespace viz
{
   class PointOrbitCameraController : public HIDInterface
   {
      public:
         PointOrbitCameraController(
            Camera & camera,
            const Vector3 & look_point,
            const float orbit_radius,
            const float orbit_frequency,
            const float orbit_height
         );

         void initialize(const data_vizConfig_t & viz_config_data);

         void mouseButton_cb(int button, int action)
         {
            if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
            {
               mouse_button_left_pressed_ = true;
            }
            if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
            {
               mouse_button_left_pressed_ = false;
            }

            if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
            {
               mouse_button_right_pressed_ = true;
            }
            if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
            {
               mouse_button_right_pressed_ = false;
            }
         }

         void mouseMove_cb(double xpos, double ypos)
         {
            // Ignore this callback if ImGui is in focus.
            ImGuiIO& io = ImGui::GetIO();

            mouse_ray_ = camera_->mouseRayCast(xpos, ypos);

            mouse_x_ = xpos;
            mouse_y_ = ypos;
         }

         void keyboard_cb(GLFWwindow * window)
         { }

         // Used to reset the camera's projection matrix when the render window
         // is resized.
         void frameBufferSize_cb(int width, int height)
         {
            camera_->updateProjection(width, height);
         }

         void windowSize_cb(int width, int height)
         {
            camera_->updateWindowDimensions(width, height);
         }

         // The state of the mouse buttons is given as output args.
         void mouseButtons(bool & left, bool & right) const
         {
            left = mouse_button_left_pressed_;
            right = mouse_button_right_pressed_;
         }

         // Updates the camera parameters after the passage of a real time
         // window, dt.
         void update(float dt);

         Vector3 & lookPoint(void)
         {
            return look_point_;
         }

         float & orbitRadius(void)
         {
            return orbit_radius_;
         }

         float & orbitFrequency(void)
         {
            return orbit_frequency_;
         }

         float & orbitHeight(void)
         {
            return orbit_height_;
         }

         const Vector3 & mouseRay(void) const
         {
            return mouse_ray_;
         }

         const Vector3 & cameraPos(void) const
         {
            return camera_pos_;
         }

      private:

         // The camera whose position and direction are modified by this
         // controller.
         Camera * camera_;

         // Point the camera is looking at in FZX coordinates.
         Vector3 look_point_;

         float orbit_radius_;

         // How many full orbits should be performed every second.
         float orbit_frequency_;

         // The z-coordinate of the orbit plane.
         float orbit_height_;

         // The angle that's modified as time is updated.
         float orbit_angle_;

         float mouse_x_;

         float mouse_y_;

         // Indicates that the left mouse button was pressed.
         bool mouse_button_left_pressed_;

         bool mouse_button_right_pressed_;

         Vector3 mouse_ray_;

         Vector3 camera_pos_;

   };
}

#endif
