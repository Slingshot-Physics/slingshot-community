#ifndef DEFAULT_CAMERA_CONTROLLER_HEADER
#define DEFAULT_CAMERA_CONTROLLER_HEADER

#include "hid_interface.hpp"

#include "camera.hpp"

#include "gl_common.hpp"

namespace viz
{

   // The default camera controller only keeps a pointer to a camera and sets
   // the camera's position, direction, and projection data. The default
   // controller uses a combination of keyboard and mouse input to update
   // internal representations of the position and pointing direction, then
   // uses those internal representations to set the camera's state.
   class DefaultCameraController : public HIDInterface
   {
      public:
         DefaultCameraController(Camera & camera);

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

         // Used to update the ray being projected from the mouse's position
         // on the screen into the rendered world.
         void windowSize_cb(int width, int height);

         // The state of the mouse buttons is given as output args.
         void mouseButtons(bool & left, bool & right) const
         {
            left = mouse_button_left_pressed_;
            right = (
               mouse_button_right_pressed_ ||
               (alt_as_rmb_ && alt_button_pressed_)
            );
         }

         float & cameraSpeed(void)
         {
            return camera_speed_;
         }

         void setCameraSpeed(float camera_speed)
         {
            camera_speed_ = camera_speed;
         }

         // Returns the unit vector direction of the ray cast by the mouse's
         // position on the screen.
         const Vector3 & mouseRay(void) const
         {
            return mouse_ray_;
         }

         const Vector3 & cameraPos(void) const
         {
            return pos_;
         }

         void setCameraPos(const Vector3 & pos)
         {
            pos_ = pos;
         }

         const Vector3 & cameraDirection(void) const
         {
            return direction_;
         }

         // Allows left-alt to register as right mouse button clicks.
         void setLeftAltAsRmb(bool alt_as_rmb)
         {
            alt_as_rmb_ = alt_as_rmb;
         }

      private:
         // Camera's yaw
         float yaw_deg_;

         // Camera's pitch
         float pitch_deg_;

         // Position in FZX coordinates.
         Vector3 pos_;

         // Camera pointing direction in FZX coordinates.
         Vector3 direction_;

         // Camera's up direction in FZX coordinates.
         Vector3 up_;

         // The camera whose position and direction are modified by this
         // controller.
         Camera * camera_;

         // Indicates that the left mouse button was pressed.
         bool mouse_button_left_pressed_;

         bool mouse_button_right_pressed_;

         bool alt_button_pressed_;

         bool last_button_state_;

         // Last known mouse X position.
         float mouseX_;

         // Last known mouse Y position.
         float mouseY_;

         float camera_speed_;

         bool alt_as_rmb_;

         Vector3 mouse_ray_;

         // Update camera direction based on the controller's yaw and pitch.
         void updateDirection(void);

         void updateMouseRay(double xpos, double ypos);

   };

}

#endif
