#include "ld52_camera_controller.hpp"

#include "imgui.h"
#include "vector3.hpp"
#include "viz_types.hpp"
#include "viztypeconverters.hpp"

#include <cmath>

namespace viz
{

   BuggyCameraController::BuggyCameraController(
      Camera & camera,
      float & motor_speed,
      float & steer_angle,
      bool & grabber_pressed,
      bool & reset_pressed
   )
      : camera_(&camera)
      , yaw_deg_(0.f)
      , pitch_deg_(0.f)
      , motor_speed_(&motor_speed)
      , steer_angle_(&steer_angle)
      , grabber_pressed_(&grabber_pressed)
      , reset_pressed_(&reset_pressed)
      , m_prev_(false)
      , grabber_key_pressed_(false)
      , grabber_mouse_pressed_(false)
      , mouse_button_left_pressed_(false)
   { }

   void BuggyCameraController::initialize(
      const data_vizConfig_t & viz_config_data
   )
   {
      viz::types::config_t viz_config;
      viz::converters::convert_data_to_vizConfig(viz_config_data, viz_config);

      const Vector3 look_direction(1.f, 0.f, 0.f);

      yaw_deg_ = -1.f * atan2(look_direction[1], look_direction[0]) * 180.f / M_PI;
      pitch_deg_ = asinf(look_direction[2]) * 180 / M_PI;

      camera_->setPos(viz_config.cameraPos);
      camera_->setLookDirection(look_direction);
      camera_->updateProjection(viz_config.windowWidth, viz_config.windowHeight);
   }

   void BuggyCameraController::mouseButton_cb(int button, int action)
   {
      // Ignore this callback if ImGui is in focus.
      ImGuiIO& io = ImGui::GetIO();

      if (io.WantCaptureMouse)
      {
         return;
      }

      if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
      {
         // *grabber_pressed_ = true;
         grabber_mouse_pressed_ = true;
      }
      if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
      {
         // *grabber_pressed_ = false;
         grabber_mouse_pressed_ = false;
      }
      *grabber_pressed_ = grabber_mouse_pressed_ || grabber_key_pressed_;

      if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
      {
         mouse_button_left_pressed_ = true;
      }
      if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
      {
         mouse_button_left_pressed_ = false;
      }
   }

   void BuggyCameraController::mouseMove_cb(double xpos, double ypos)
   {
      // Ignore this callback if ImGui is in focus.
      ImGuiIO& io = ImGui::GetIO();

      if (io.WantCaptureMouse)
      {
         return;
      }

      static bool last_button_state = false;
      if (!mouse_button_left_pressed_)
      {
         last_button_state = mouse_button_left_pressed_;
         return;
      }

      if (mouse_button_left_pressed_ != last_button_state)
      {
         mouseX_ = xpos;
         mouseY_ = ypos;
      }

      const float sensitivity = 0.1f;
      float xOffset = (xpos - mouseX_) * sensitivity;
      float yOffset = (mouseY_ - ypos) * sensitivity;
      mouseX_ = xpos;
      mouseY_ = ypos;

      yaw_deg_ += xOffset;
      pitch_deg_ += yOffset;

      clampYawPitch();

      last_button_state = mouse_button_left_pressed_;
   }

   void BuggyCameraController::keyboard_cb(GLFWwindow * window)
   {
      if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
      {
         *motor_speed_ = 40.f;
      }
      if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
      {
         *motor_speed_ = -40.f;
      }
      if (
         (glfwGetKey(window, GLFW_KEY_S) == GLFW_RELEASE) &&
         (glfwGetKey(window, GLFW_KEY_W) == GLFW_RELEASE)
      )
      {
         *motor_speed_ = 0.f;
      }

      if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
      {
         if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
         {
            *steer_angle_ = 10.f * M_PI / 180.f;
         }
         else
         {
            *steer_angle_ = 15.f * M_PI / 180.f;
         }
      }
      if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
      {
         if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
         {
            *steer_angle_ = -10.f * M_PI / 180.f;
         }
         else
         {
            *steer_angle_ = -15.f * M_PI / 180.f;
         }
      }
      if (
         (glfwGetKey(window, GLFW_KEY_D) == GLFW_RELEASE) &&
         (glfwGetKey(window, GLFW_KEY_A) == GLFW_RELEASE)
      )
      {
         *steer_angle_ = 0.f;
      }

      if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
      {
         // *grabber_pressed_ = true;
         grabber_key_pressed_ = true;
      }
      if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
      {
         // *grabber_pressed_ = false;
         grabber_key_pressed_ = false;
      }
      *grabber_pressed_ = grabber_mouse_pressed_ || grabber_key_pressed_;

      if (
         m_prev_ && 
         (glfwGetKey(window, GLFW_KEY_M) == GLFW_RELEASE)
      )
      {
         *reset_pressed_ = true;
      }
      else
      {
         *reset_pressed_ = false;
      }

      m_prev_ = (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS);
   }

   void BuggyCameraController::frameBufferSize_cb(int width, int height)
   {
      camera_->updateProjection(width, height);
   }

   void BuggyCameraController::getYawPitchDeg(float & yaw_deg, float & pitch_deg)
   {
      yaw_deg = yaw_deg_;
      pitch_deg = pitch_deg_;
   }

   void BuggyCameraController::clampYawPitch(void)
   {
      if (pitch_deg_ > 5.0f)
      {
         pitch_deg_ = 5.0f;
      }
      if (pitch_deg_ < -45.0f)
      {
         pitch_deg_ = -45.0f;
      }
      if (yaw_deg_ > 135.f)
      {
         yaw_deg_ = 135.f;
      }
      if (yaw_deg_ < -135.f)
      {
         yaw_deg_ = -135.f;
      }
   }
}
