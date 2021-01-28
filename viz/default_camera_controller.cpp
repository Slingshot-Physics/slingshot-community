#include "default_camera_controller.hpp"

#include "imgui.h"
#include "vector3.hpp"
#include "viz_types.hpp"
#include "viztypeconverters.hpp"

#include <cmath>

namespace viz
{

   DefaultCameraController::DefaultCameraController(Camera & camera)
      : yaw_deg_(0.f)
      , pitch_deg_(0.f)
      , camera_(&camera)
      , pos_(0.f, 0.f, 0.f)
      , direction_(0.f, 1.f, 0.f)
      , up_(0.f, 0.f, 1.f)
      , mouse_button_left_pressed_(false)
      , mouse_button_right_pressed_(false)
      , alt_button_pressed_(false)
      , last_button_state_(false)
      , camera_speed_(0.15f)
      , alt_as_rmb_(false)
   {
      camera_->setPos(pos_);
      camera_->setLookDirection(direction_);
      camera_->updateProjection(800, 600);
   }

   void DefaultCameraController::initialize(
      const data_vizConfig_t & viz_config_data
   )
   {
      viz::types::config_t viz_config;
      viz::converters::convert_data_to_vizConfig(viz_config_data, viz_config);
      pos_ = viz_config.cameraPos;
      direction_ = (viz_config.cameraPoint - viz_config.cameraPos).unitVector();

      yaw_deg_ = -1.f * atan2(direction_[1], direction_[0]) * 180.f / M_PI;
      pitch_deg_ = asinf(direction_[2]) * 180 / M_PI;

      camera_->setPos(pos_);
      camera_->setLookDirection(direction_);
      camera_->updateProjection(viz_config.windowWidth, viz_config.windowHeight);
   }

   void DefaultCameraController::mouseButton_cb(int button, int action)
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

   void DefaultCameraController::mouseMove_cb(double xpos, double ypos)
   {
      // Ignore this callback if ImGui is in focus.
      ImGuiIO& io = ImGui::GetIO();

      updateMouseRay(xpos, ypos);

      if (io.WantCaptureMouse)
      {
         return;
      }

      if (!mouse_button_left_pressed_)
      {
         last_button_state_ = mouse_button_left_pressed_;
         return;
      }

      if (mouse_button_left_pressed_ != last_button_state_)
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

      updateDirection();

      last_button_state_ = mouse_button_left_pressed_;
   }

   void DefaultCameraController::keyboard_cb(GLFWwindow * window)
   {
      // Ignore this callback if ImGui is in focus.
      ImGuiIO& io = ImGui::GetIO();

      if (io.WantCaptureKeyboard)
      {
         return;
      }

      if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
      {
         Vector3 forward = direction_;
         forward[2] = 0.f;
         pos_ += camera_speed_ * forward;
         camera_->setPos(pos_);
      }
      if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
      {
         Vector3 forward = direction_;
         forward[2] = 0.f;
         pos_ -= camera_speed_ * forward;
         camera_->setPos(pos_);
      }
      if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
      {
         pos_ -= direction_.crossProduct(up_).unitVector() * camera_speed_;
         camera_->setPos(pos_);
      }
      if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
      {
         pos_ += direction_.crossProduct(up_).unitVector() * camera_speed_;
         camera_->setPos(pos_);
      }
      if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
      {
         pos_ += camera_speed_ * Vector3(0.f, 0.f, 1.f) * 0.5f;
         camera_->setPos(pos_);
      }
      if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
      {
         pos_ -= camera_speed_ * Vector3(0.f, 0.f, 1.f) * 0.5f;
         camera_->setPos(pos_);
      }
      if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
      {
         pitch_deg_ += camera_speed_;
      }
      if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
      {
         pitch_deg_ -= camera_speed_;
      }
      if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
      {
         yaw_deg_ -= camera_speed_;
      }
      if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
      {
         yaw_deg_ += camera_speed_;
      }

      if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
      {
         alt_button_pressed_ = true;
      }
      if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_RELEASE)
      {
         alt_button_pressed_ = false;
      }

      updateDirection();
   }

   void DefaultCameraController::frameBufferSize_cb(int width, int height)
   {
      camera_->updateProjection(width, height);
   }

   void DefaultCameraController::windowSize_cb(int width, int height)
   {
      camera_->updateWindowDimensions(width, height);
   }

   void DefaultCameraController::updateDirection(void)
   {
      if (pitch_deg_ > 89.0f)
      {
         pitch_deg_ = 89.0f;
      }
      if (pitch_deg_ < -89.0f)
      {
         pitch_deg_ = -89.0f;
      }

      float yaw_rad = yaw_deg_ * M_PI / 180.f;
      float pitch_rad = pitch_deg_ * M_PI / 180.f;

      // Spherical coordinates, ish, where zero pitch and zero yaw point
      // straight down the x-axis, and where yaw increase and decrease is
      // left-handed.
      direction_[0] = cos(pitch_rad) * cos(-1.f * yaw_rad);
      direction_[1] = cos(pitch_rad) * sin(-1.f * yaw_rad);
      direction_[2] = sin(pitch_rad);
      camera_->setLookDirection(direction_);
   }

   void DefaultCameraController::updateMouseRay(double xpos, double ypos)
   {
      mouse_ray_ = camera_->mouseRayCast(xpos, ypos);
   }

}
