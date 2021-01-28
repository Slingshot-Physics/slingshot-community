#ifndef SCUFFED_LEAGUE_CALLBACK_HEADER
#define SCUFFED_LEAGUE_CALLBACK_HEADER

#include "slingshot_callback_base.hpp"

#include "scuffed_camera_controller.hpp"
#include "scuffed_gui.hpp"

#include <deque>
#include <map>
#include <vector>

// Applies a force over the bump duration to the base of the cart.
class ScuffedCallback : public slingshot::CallbackBase
{
   public:
      ScuffedCallback(void)
         : count_(0)
         , score_(0.f)
         , bucket_box_{
            {3.f, 3.f, 8.1f}, {-3.f, -3.f, 7.1f}, 
         }
         , camera_offset_(0.f, -5.f, 4.f)
         , camera_controller_(
            camera_,
            motor_speed_,
            steer_angle_,
            reset_pressed_
         )
         , reset_pressed_(false)
         , motor_speed_(0.f)
         , k_steering_(5.f)
         , steer_angle_(0.f)
         , orbit_radius_(5.f)
         , bl_drive_motor_(nullptr)
         , br_drive_motor_(nullptr)
         , fl_steer_motor_(nullptr)
         , fr_steer_motor_(nullptr)
         , buggy_base_(nullptr)
         , fl_wheel_(nullptr)
         , fr_wheel_(nullptr)
         , bl_wheel_(nullptr)
         , br_wheel_(nullptr)
         , fl_axle_(nullptr)
         , fr_axle_(nullptr)
         , bl_axle_(nullptr)
         , br_axle_(nullptr)
      { }

      void post_setup(oy::Handle & handle);

      bool operator()(oy::Handle & handle);

      viz::HIDInterface & hid(void);

      viz::Camera & camera(void);

      void update_camera(void);

      void parse_gui(
         viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
      );

      viz::GuiCallbackBase * gui(void)
      {
         return &gui_;
      }

   private:
      // Number of sim updates
      unsigned int count_;

      float score_;

      ScuffedGui gui_;

      geometry::types::aabb_t bucket_box_;

      Vector3 camera_offset_;

      viz::Camera camera_;

      viz::ScuffedCameraController camera_controller_;

      bool reset_pressed_;

      float motor_speed_;

      float k_steering_;

      float steer_angle_;

      float orbit_radius_;

      trecs::uid_t ball_id_;

      trecs::uid_t buggy_base_id_;

      trecs::uid_t fl_wheel_id_;

      trecs::uid_t fr_wheel_id_;

      trecs::uid_t bl_wheel_id_;

      trecs::uid_t br_wheel_id_;

      trecs::uid_t fl_axle_id_;

      trecs::uid_t fr_axle_id_;

      trecs::uid_t bl_axle_id_;

      trecs::uid_t br_axle_id_;

      oy::types::constraintRevoluteMotor_t * bl_drive_motor_;

      oy::types::constraintRevoluteMotor_t * br_drive_motor_;

      oy::types::constraintRevoluteMotor_t * fl_steer_motor_;

      oy::types::constraintRevoluteMotor_t * fr_steer_motor_;

      oy::types::rigidBody_t * ball_;

      oy::types::rigidBody_t * buggy_base_;

      oy::types::rigidBody_t * fl_wheel_;

      oy::types::rigidBody_t * fr_wheel_;

      oy::types::rigidBody_t * bl_wheel_;

      oy::types::rigidBody_t * br_wheel_;

      oy::types::rigidBody_t * fl_axle_;

      oy::types::rigidBody_t * fr_axle_;

      oy::types::rigidBody_t * bl_axle_;

      oy::types::rigidBody_t * br_axle_;

      std::map<trecs::uid_t, oy::types::rigidBody_t> initial_buggy_states_;

      // Puts the player in the exact same position they started in.
      void respawn_player(oy::Handle & handle);

};

#endif
