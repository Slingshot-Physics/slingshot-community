#include "point_orbit_camera_controller.hpp"

#include <cmath>

namespace viz
{

   PointOrbitCameraController::PointOrbitCameraController(
      Camera & camera,
      const Vector3 & look_point,
      const float orbit_radius,
      const float orbit_frequency,
      const float orbit_height
   )
      : camera_(&camera)
      , look_point_(look_point)
      , orbit_radius_(orbit_radius)
      , orbit_frequency_(orbit_frequency)
      , orbit_angle_(0.f)
      , orbit_height_(orbit_height)
      , mouse_button_left_pressed_(false)
      , mouse_button_right_pressed_(false)
   { }

   void PointOrbitCameraController::initialize(const data_vizConfig_t & viz_config_data)
   {
      viz::types::config_t viz_config;
      viz::converters::convert_data_to_vizConfig(viz_config_data, viz_config);

      const Vector3 look_direction = look_point_ - viz_config.cameraPos;

      camera_->setPos(viz_config.cameraPos);
      camera_->setLookDirection(look_direction);
      camera_->updateProjection(viz_config.windowWidth, viz_config.windowHeight);
   }

   void PointOrbitCameraController::update(float dt)
   {
      orbit_angle_ += M_PI * 2.f * orbit_frequency_ * dt;

      orbit_angle_ -= (orbit_angle_ > 2.f * M_PI) ? 2.f * M_PI : 0.f;

      camera_pos_.Initialize(
         orbit_radius_ * cosf(orbit_angle_) + look_point_[0],
         orbit_radius_ * sinf(orbit_angle_) + look_point_[1],
         orbit_height_
      );

      const Vector3 look_direction(
         look_point_ - camera_pos_
      );

      camera_->setPos(camera_pos_);
      camera_->setLookDirection(look_direction);

      mouse_ray_ = camera_->mouseRayCast(mouse_x_, mouse_y_);
   }
}
