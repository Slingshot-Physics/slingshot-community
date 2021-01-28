#ifndef LD52_CAMERA_CONTROLLER_HEADER
#define LD52_CAMERA_CONTROLLER_HEADER

#include "hid_interface.hpp"

#include "camera.hpp"

#include "gl_common.hpp"

namespace viz
{

   class BuggyCameraController : public HIDInterface
   {
      public:
         BuggyCameraController(
            Camera & camera,
            float & motor_speed,
            float & steer_angle,
            bool & grabber_pressed,
            bool & reset_pressed
         );
      
         void initialize(const data_vizConfig_t & viz_config_data);

         // Look for left mouse button clicks.
         void mouseButton_cb(int button, int action);

         // Use mouse position changes to change the camera's look-at point.
         void mouseMove_cb(double xpos, double ypos);

         // Process the last known keyboard states to change the camera's
         // position and look-at point.
         void keyboard_cb(GLFWwindow * window);

         // Used to reset the camera's projection matrix when the render window
         // is resized.
         void frameBufferSize_cb(int width, int height);

         void windowSize_cb(int width, int height)
         {
            camera_->updateWindowDimensions(width, height);
         }

         void getYawPitchDeg(float & yaw_deg, float & pitch_deg);

      private:
         // The camera whose position and direction are modified by this
         // controller.
         Camera * camera_;

         // Camera's yaw
         float yaw_deg_;

         // Camera's pitch
         float pitch_deg_;

         // Sets the revolute motor constraint's angular speed.
         float * motor_speed_;

         float * steer_angle_;

         bool * grabber_pressed_;

         bool * reset_pressed_;

         bool m_prev_;

         bool grabber_key_pressed_;

         bool grabber_mouse_pressed_;

         // Indicates that the left mouse button was pressed.
         bool mouse_button_left_pressed_;

         // Last known mouse X position.
         float mouseX_;

         // Last known mouse Y position.
         float mouseY_;

         // Update camera position based on the controller's yaw and pitch.
         void clampYawPitch(void);
   };

}

#endif
