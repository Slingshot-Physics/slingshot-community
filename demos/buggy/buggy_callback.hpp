#ifndef CART_CALLBACK_HEADER
#define CART_CALLBACK_HEADER

#include "slingshot_callback_base.hpp"

#include "buggy_camera_controller.hpp"

// Applies a force over the bump duration to the base of the cart.
class BuggyCallback : public slingshot::CallbackBase
{
   public:
      BuggyCallback(void)
         : count_(0)
         , camera_offset_(0.f, -5.f, 4.f)
         , camera_controller_(camera_, motor_speed_, steer_angle_)
         , motor_speed_(0.f)
         , k_steering_(5.f)
         , steer_angle_(0.f)
         , orbit_radius_(5.f)
         , drive_motor_(nullptr)
         , fl_steer_motor_(nullptr)
         , fr_steer_motor_(nullptr)
         , buggy_base_(nullptr)
         , fl_wheel_(nullptr)
         , fr_wheel_(nullptr)
         , bl_wheel_(nullptr)
         , br_wheel_(nullptr)
         , fl_axle_(nullptr)
         , fr_axle_(nullptr)
      { }

      void post_setup(oy::Handle & handle);

      bool operator()(oy::Handle & handle);

      viz::HIDInterface & hid(void);

      viz::Camera & camera(void);

      void update_camera(void);

   private:
      // Number of sim updates
      unsigned int count_;

      Vector3 camera_offset_;

      viz::Camera camera_;

      viz::BuggyCameraController camera_controller_;

      float motor_speed_;

      float k_steering_;

      float steer_angle_;

      float orbit_radius_;

      int buggy_base_id_;

      int fl_wheel_id_;

      int fr_wheel_id_;

      int bl_wheel_id_;

      int br_wheel_id_;

      int fl_axle_id_;

      int fr_axle_id_;

      oy::types::constraintRevoluteMotor_t * drive_motor_;

      oy::types::constraintRevoluteMotor_t * fl_steer_motor_;

      oy::types::constraintRevoluteMotor_t * fr_steer_motor_;

      oy::types::rigidBody_t * buggy_base_;

      oy::types::rigidBody_t * fl_wheel_;

      oy::types::rigidBody_t * fr_wheel_;

      oy::types::rigidBody_t * bl_wheel_;

      oy::types::rigidBody_t * br_wheel_;

      oy::types::rigidBody_t * fl_axle_;

      oy::types::rigidBody_t * fr_axle_;
};

#endif
