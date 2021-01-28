#ifndef QUADRUPED_CALLBACK_HEADER
#define QUADRUPED_CALLBACK_HEADER

#include "default_camera_controller.hpp"
#include "slingshot_types.hpp"
#include "quadruped_gui.hpp"

#include "slingshot_callback_base.hpp"

#include <array>
#include <map>
#include <string>

float calculateRevoluteMotorTheta(
   oy::types::rigidBody_t body_a,
   oy::types::rigidBody_t body_b,
   int angle_axis_index,
   int orthogonal_axis_index,
   float sign
);

class QuadrupedCallback : public slingshot::CallbackBase
{
   public:
      QuadrupedCallback(void);

      ~QuadrupedCallback(void)
      { }

      void post_setup(oy::Handle & handle);

      bool operator()(oy::Handle & handle);

      void parse_gui(
         viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
      );

      viz::GuiCallbackBase * gui(void);

      viz::HIDInterface & hid(void);

      viz::Camera & camera(void);

   private:

      bool run_sim_;

      bool step_one_;

      bool rmb_hold_;

      bool rmb_release_;

      bool rmb_change_;

      trecs::uid_t grabbed_body_uid_;

      trecs::uid_t spring_grabber_uid_;

      trecs::uid_t damper_grabber_uid_;

      Vector3 camera_pos_;

      Vector3 camera_ray_slope_;

      float grab_dist_;

      QuadrupedGuiCallback gui_;

      viz::DefaultCameraController camera_controller_;

      viz::Camera camera_;

      std::map<std::string, int> body_name_to_scenario_id_;

      // std::map<std::string, int> motor_name_to_scenario_id_;

      std::map<std::string, oy::types::rigidBody_t *> body_name_to_body_;

      std::map<std::string, oy::types::constraintRevoluteMotor_t *> motor_name_to_motor_;

      std::array<oy::types::constraintRevoluteMotor_t *, 12> motors_;

      std::array<oy::types::rigidBody_t *, 13> bodies_;

      float hip_rotation_angle_setpoint_;

      float hip_flexion_angle_setpoint_;

      float knee_flexion_angle_setpoint_;

      void setMotorSpeeds(void);
};

#endif
