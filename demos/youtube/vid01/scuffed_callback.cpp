#include "scuffed_callback.hpp"

#include "random_utils.hpp"

#include <chrono>
#include <unordered_set>
#include <thread>
#include <iostream>

void ScuffedCallback::post_setup(oy::Handle & handle)
{
   ball_id_ = handle.getIdMapping().at(87);
   buggy_base_id_ = handle.getIdMapping().at(2);

   fl_wheel_id_ = handle.getIdMapping().at(4);
   fr_wheel_id_ = handle.getIdMapping().at(6);
   bl_wheel_id_ = handle.getIdMapping().at(8);
   br_wheel_id_ = handle.getIdMapping().at(10);

   fl_axle_id_ = handle.getIdMapping().at(12);
   fr_axle_id_ = handle.getIdMapping().at(14);
   bl_axle_id_ = handle.getIdMapping().at(16);
   br_axle_id_ = handle.getIdMapping().at(18);

   ball_ = &(handle.getBody(ball_id_));
   buggy_base_ = &(handle.getBody(buggy_base_id_));
   fl_wheel_ = &(handle.getBody(fl_wheel_id_));
   fr_wheel_ = &(handle.getBody(fr_wheel_id_));
   bl_wheel_ = &(handle.getBody(bl_wheel_id_));
   br_wheel_ = &(handle.getBody(br_wheel_id_));
   fl_axle_ = &(handle.getBody(fl_axle_id_));
   fr_axle_ = &(handle.getBody(fr_axle_id_));
   bl_axle_ = &(handle.getBody(bl_axle_id_));
   br_axle_ = &(handle.getBody(br_axle_id_));

   initial_buggy_states_[buggy_base_id_] = *buggy_base_;
   initial_buggy_states_[fl_wheel_id_] = *fl_wheel_;
   initial_buggy_states_[fr_wheel_id_] = *fr_wheel_;
   initial_buggy_states_[bl_wheel_id_] = *bl_wheel_;
   initial_buggy_states_[br_wheel_id_] = *br_wheel_;
   initial_buggy_states_[fl_axle_id_] = *fl_axle_;
   initial_buggy_states_[fr_axle_id_] = *fr_axle_;
   initial_buggy_states_[bl_axle_id_] = *bl_axle_;
   initial_buggy_states_[br_axle_id_] = *br_axle_;

   const auto & revolute_motor_uids = handle.getRevoluteMotorConstraintUids();

   for (const auto revolute_motor_uid : revolute_motor_uids)
   {
      oy::types::constraintRevoluteMotor_t & temp_motor = handle.getRevoluteMotorConstraint(revolute_motor_uid);
      oy::types::bodyLink_t motor_link = handle.getBodyLink(revolute_motor_uid);
      if (
         (motor_link.parentId == bl_wheel_id_ && motor_link.childId == bl_axle_id_) ||
         (motor_link.parentId == bl_axle_id_ && motor_link.childId == bl_wheel_id_)
      )
      {
         bl_drive_motor_ = &temp_motor;
      }

      if (
         (motor_link.parentId == br_wheel_id_ && motor_link.childId == br_axle_id_) ||
         (motor_link.parentId == br_axle_id_ && motor_link.childId == br_wheel_id_)
      )
      {
         br_drive_motor_ = &temp_motor;
      }

      if (
         (motor_link.parentId == buggy_base_id_ && motor_link.childId == fl_axle_id_) ||
         (motor_link.parentId == fl_axle_id_ && motor_link.childId == buggy_base_id_)
      )
      {
         fl_steer_motor_ = &temp_motor;
      }

      if (
         (motor_link.parentId == buggy_base_id_ && motor_link.childId == fr_axle_id_) ||
         (motor_link.parentId == fr_axle_id_ && motor_link.childId == buggy_base_id_)
      )
      {
         fr_steer_motor_ = &temp_motor;
      }
   }
}

bool ScuffedCallback::operator()(oy::Handle & handle)
{
   Matrix33 R_W_to_Buggy = buggy_base_->ql2b.rotationMatrix();
   Matrix33 R_W_to_FLA = fl_axle_->ql2b.rotationMatrix();
   Matrix33 R_W_to_FRA = fr_axle_->ql2b.rotationMatrix();

   Matrix33 R_FLA_to_Buggy = R_W_to_Buggy * R_W_to_FLA.transpose();
   Matrix33 R_FRA_to_Buggy = R_W_to_Buggy * R_W_to_FRA.transpose();

   Vector3 x_hat(1.f, 0.f, 0.f);
   Vector3 y_hat(0.f, 1.f, 0.f);

   float fl_axle_error = -1.f * sinf(steer_angle_) - (R_FLA_to_Buggy * y_hat)[0];

   float fr_axle_error = -1.f * sinf(steer_angle_) - (R_FRA_to_Buggy * y_hat)[0];

   if (fl_steer_motor_ != nullptr)
   {
      fl_steer_motor_->angularSpeed = -1.f * k_steering_ * fl_axle_error;
   }

   if (fr_steer_motor_ != nullptr)
   {
      fr_steer_motor_->angularSpeed = k_steering_ * fr_axle_error;
   }

   if (bl_drive_motor_ != nullptr)
   {
      bl_drive_motor_->angularSpeed = motor_speed_ - steer_angle_ * motor_speed_ * 0.5f;
   }

   if (br_drive_motor_ != nullptr)
   {
      br_drive_motor_->angularSpeed = motor_speed_ + steer_angle_ * motor_speed_ * 0.5f;
   }

   count_ += 1;

   update_camera();

   if (buggy_base_->linPos[2] < -64.f || reset_pressed_)
   {
      respawn_player(handle);
   }

   if (ball_->linPos[2] < -64.f)
   {
      ball_->linPos[0] = 0.f;
      ball_->linPos[1] = 0.f;
      ball_->linPos[2] = 12.f;
      ball_->linVel[0] = 0.f;
      ball_->linVel[1] = 0.f;
      ball_->linVel[2] = 0.f;
   }

   if (count_ % 100 == 0)
   {
      std::cout << "num sim steps: " << count_ << std::endl;
   }

   // Roll it!
   return true;
}

viz::HIDInterface & ScuffedCallback::hid(void)
{
   viz::HIDInterface & hid_controller = camera_controller_;
   return hid_controller;
}

viz::Camera & ScuffedCallback::camera(void)
{
   return camera_;
}

void ScuffedCallback::update_camera(void)
{
   float yaw_deg, pitch_deg;
   camera_controller_.getYawPitchDeg(yaw_deg, pitch_deg);

   float yaw_rad = yaw_deg * M_PI/180.f;
   float pitch_rad = pitch_deg * M_PI/180.f;

   // Elliptical orbit - farther away if you're looking straight down, closer
   // if you're looking at the car laterally.
   Vector3 orbit_pos_body(
      orbit_radius_ * 2.f * cos(-1.f * pitch_rad) * sin(-1.f * yaw_rad),
      orbit_radius_  * 2.f * -1.f * cos(-1.f * pitch_rad) * cos(-1.f * yaw_rad),
      orbit_radius_ * 3.f * sin(-1.f * pitch_rad)
   );

   Matrix33 R_Buggy_to_W = buggy_base_->ql2b.rotationMatrix().transpose();
   Vector3 orbit_pos_W(R_Buggy_to_W * orbit_pos_body + buggy_base_->linPos);

   Vector3 look_direction(buggy_base_->linPos - orbit_pos_W);
   look_direction.Normalize();

   camera_.setPos(orbit_pos_W);
   camera_.setLookDirection(look_direction);

}

void ScuffedCallback::parse_gui(
   viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
)
{
   if (count_ % 10 != 0)
   {
      return;
   }

   ScuffedGuiState_t & gui_state = gui_.getState();

   gui_state.score = score_;
}

void ScuffedCallback::respawn_player(oy::Handle & handle)
{
   for (const auto sim_id_to_state : initial_buggy_states_)
   {
      handle.getBody(sim_id_to_state.first) = sim_id_to_state.second;
   }
}
