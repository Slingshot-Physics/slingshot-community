#include "quadruped_callback.hpp"

#include "raycast.hpp"
#include "transform.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

static float clamp(float val, float min, float max)
{
   return std::max(std::min(val, max), min);
}

float calculateRevoluteMotorTheta(
   oy::types::rigidBody_t body_a,
   oy::types::rigidBody_t body_b,
   int angle_axis_index,
   int orthogonal_axis_index,
   float sign
)
{
   const Vector3 e_hat[3] = {
      {1.f, 0.f, 0.f},
      {0.f, 1.f, 0.f},
      {0.f, 0.f, 1.f}
   };

   const Matrix33 R_W_to_A(body_a.ql2b.rotationMatrix());
   const Matrix33 R_W_to_B(body_b.ql2b.rotationMatrix());
   const Matrix33 R_A_to_B(R_W_to_B * R_W_to_A.transpose());

   const Vector3 e_hat_B = R_A_to_B * e_hat[angle_axis_index];

   float cos_theta_b = e_hat_B.dot(e_hat[angle_axis_index]);
   float sin_theta_b = sign * (
      e_hat_B.crossProduct(e_hat[angle_axis_index])
   ).dot(
      e_hat[orthogonal_axis_index]
   );

   float theta_b = std::atan2(
      clamp(sin_theta_b, -1.f, 1.f), clamp(cos_theta_b, -1.f, 1.f)
   );

   return theta_b;
}

QuadrupedCallback::QuadrupedCallback(void)
   : run_sim_(true)
   , step_one_(false)
   , rmb_hold_(false)
   , rmb_release_(true)
   , rmb_change_(false)
   , grabbed_body_uid_(-1)
   , spring_grabber_uid_(-1)
   , damper_grabber_uid_(-1)
   , grab_dist_(0.f)
   , camera_controller_(camera_)
   , hip_rotation_angle_setpoint_(0.f)
   , hip_flexion_angle_setpoint_(0.f)
   , knee_flexion_angle_setpoint_(0.f)
{
   camera_controller_.setLeftAltAsRmb(true);
}

void QuadrupedCallback::post_setup(oy::Handle & handle)
{
   body_name_to_scenario_id_["base"] = 1;
   body_name_to_scenario_id_["bl_thigh"] = 2;
   body_name_to_scenario_id_["bl_calf"] = 3;
   body_name_to_scenario_id_["fl_thigh"] = 4;
   body_name_to_scenario_id_["fl_calf"] = 5;
   body_name_to_scenario_id_["fr_thigh"] = 6;
   body_name_to_scenario_id_["fr_calf"] = 7;
   body_name_to_scenario_id_["br_thigh"] = 8;
   body_name_to_scenario_id_["br_calf"] = 9;
   body_name_to_scenario_id_["bl_hip"] = 10;
   body_name_to_scenario_id_["fl_hip"] = 11;
   body_name_to_scenario_id_["fr_hip"] = 12;
   body_name_to_scenario_id_["br_hip"] = 13;

   std::map<std::string, trecs::uid_t> body_name_to_sim_id;

   for (const auto & name_id : body_name_to_scenario_id_)
   {
      body_name_to_body_[name_id.first] = \
         &handle.getBody(handle.getIdMapping().at(name_id.second));

      body_name_to_sim_id[name_id.first] = \
         handle.getIdMapping().at(name_id.second);
   }

   struct body_id_pair_t
   {
      trecs::uid_t idA;
      trecs::uid_t idB;
   };

   body_id_pair_t body_ids;

   std::map<std::string, body_id_pair_t> motor_name_to_linked_bodies;

   motor_name_to_linked_bodies["bl_hip"] = {
      body_name_to_sim_id["base"],
      body_name_to_sim_id["bl_hip"]
   };

   motor_name_to_linked_bodies["fl_hip"] = {
      body_name_to_sim_id["base"],
      body_name_to_sim_id["fl_hip"]
   };

   motor_name_to_linked_bodies["fr_hip"] = {
      body_name_to_sim_id["base"],
      body_name_to_sim_id["fr_hip"]
   };

   motor_name_to_linked_bodies["br_hip"] = {
      body_name_to_sim_id["base"],
      body_name_to_sim_id["br_hip"]
   };

   motor_name_to_linked_bodies["bl_thigh"] = {
      body_name_to_sim_id["bl_hip"],
      body_name_to_sim_id["bl_thigh"]
   };

   motor_name_to_linked_bodies["fl_thigh"] = {
      body_name_to_sim_id["fl_hip"],
      body_name_to_sim_id["fl_thigh"]
   };

   motor_name_to_linked_bodies["fr_thigh"] = {
      body_name_to_sim_id["fr_hip"],
      body_name_to_sim_id["fr_thigh"]
   };

   motor_name_to_linked_bodies["br_thigh"] = {
      body_name_to_sim_id["br_hip"],
      body_name_to_sim_id["br_thigh"]
   };

   motor_name_to_linked_bodies["bl_calf"] = {
      body_name_to_sim_id["bl_thigh"],
      body_name_to_sim_id["bl_calf"]
   };

   motor_name_to_linked_bodies["fl_calf"] = {
      body_name_to_sim_id["fl_thigh"],
      body_name_to_sim_id["fl_calf"]
   };

   motor_name_to_linked_bodies["fr_calf"] = {
      body_name_to_sim_id["fr_thigh"],
      body_name_to_sim_id["fr_calf"]
   };

   motor_name_to_linked_bodies["br_calf"] = {
      body_name_to_sim_id["br_thigh"],
      body_name_to_sim_id["br_calf"]
   };

   const auto & motor_uids = handle.getRevoluteMotorConstraintUids();
   for (const auto name_to_body_ids : motor_name_to_linked_bodies)
   {
      for (const auto motor_uid : motor_uids)
      {
         auto & motor = handle.getRevoluteMotorConstraint(motor_uid);
         oy::types::bodyLink_t motor_link = handle.getBodyLink(motor_uid);
         if (
            (
               (motor_link.parentId == name_to_body_ids.second.idA) &&
               (motor_link.childId == name_to_body_ids.second.idB)
            ) ||
            (
               (motor_link.parentId == name_to_body_ids.second.idB) &&
               (motor_link.childId == name_to_body_ids.second.idA)
            )
         )
         {
            std::cout << "found a motor match for " << name_to_body_ids.first << " at uid " << motor_uid << "\n";
            motor_name_to_motor_[name_to_body_ids.first] = &motor;
            break;
         }
      }
   }
}

bool QuadrupedCallback::operator()(oy::Handle & handle)
{
   if (!run_sim_ && !step_one_)
   {
      return false;
   }

   if (step_one_)
   {
      step_one_ = false;
      run_sim_ = false;
   }

   if (rmb_change_ && rmb_release_ && grabbed_body_uid_ >= 0)
   {
      handle.removeEntity(spring_grabber_uid_);
      handle.removeEntity(damper_grabber_uid_);
      spring_grabber_uid_ = -1;
      grabbed_body_uid_ = -1;
   }
   else if (rmb_change_ && rmb_hold_ && spring_grabber_uid_ < 0)
   {
      oy::types::raycastResult_t result = handle.raycast(
         camera_pos_, camera_ray_slope_, 30.f
      );

      if (result.hit)
      {
         geometry::types::transform_t trans_Bo_to_W = handle.getBodyTransform(result.bodyId);

         grab_dist_ = (result.hits[0] - camera_pos_).magnitude();

         oy::types::forceSpring_t temp_spring;
         temp_spring.parentLinkPoint = geometry::transform::inverseBound(
            trans_Bo_to_W, result.hits[0]
         );
         temp_spring.childLinkPoint = camera_pos_ + camera_ray_slope_ * grab_dist_;
         temp_spring.restLength = 0.f;
         temp_spring.springCoeff = -75.f;

         spring_grabber_uid_ = handle.addSpringForce(result.bodyId, -1, temp_spring);

         oy::types::forceVelocityDamper_t temp_damper;
         temp_damper.damperCoeff = -5.f;
         temp_damper.parentLinkPoint = temp_spring.parentLinkPoint;
         temp_damper.childLinkPoint = temp_spring.childLinkPoint;

         damper_grabber_uid_ = handle.addVelocityDamperForce(result.bodyId, -1, temp_damper);

         grabbed_body_uid_ = static_cast<trecs::uid_t>(result.bodyId);
      }
   }

   if (spring_grabber_uid_ >= 0)
   {
      oy::types::forceSpring_t & temp_spring = \
         handle.getSpringForce(spring_grabber_uid_);
      temp_spring.childLinkPoint = camera_pos_ + camera_ray_slope_ * grab_dist_;

      oy::types::forceVelocityDamper_t & temp_damper = \
         handle.getVelocityDamperForce(damper_grabber_uid_);
      temp_damper.childLinkPoint = camera_pos_ + camera_ray_slope_ * grab_dist_;
   }

   setMotorSpeeds();

   // Roll it!
   return true;
}

void QuadrupedCallback::parse_gui(
   viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
)
{
   (void)fzx_to_viz_ids;
   QuadrupedGuiState_t & gui_state = gui_.getState();

   camera_controller_.cameraSpeed() = gui_state.camera_speed;
   run_sim_ = gui_state.run_sim;
   step_one_ = gui_state.step_one;

   gui_state.num_meshes = renderer->numMeshes();
   gui_state.num_draw_calls = renderer->numDrawCalls();
   gui_state.camera_direction = camera_controller_.cameraDirection();

   gui_state.show_grid ? renderer->enableGrid() : renderer->disableGrid();

   bool rmb_press = false;
   bool lmb_press = false;

   camera_controller_.mouseButtons(lmb_press, rmb_press);

   rmb_change_ = (rmb_press != rmb_hold_);
   if (rmb_change_)
   {
      rmb_hold_ = rmb_press;
      rmb_release_ = !rmb_press;
   }

   camera_pos_ = camera_controller_.cameraPos();
   camera_ray_slope_ = camera_controller_.mouseRay();

   renderer->setLightDirection(gui_state.light_direction);

   hip_rotation_angle_setpoint_ = gui_state.hip_rotation_angle_setpoint;

   hip_flexion_angle_setpoint_ = gui_state.hip_flexion_angle_setpoint;

   knee_flexion_angle_setpoint_ = gui_state.knee_flexion_angle_setpoint;
}

viz::GuiCallbackBase * QuadrupedCallback::gui(void)
{
   return &gui_;
}

viz::HIDInterface & QuadrupedCallback::hid(void)
{
   viz::HIDInterface & hid_controller = camera_controller_;
   return hid_controller;
}

viz::Camera & QuadrupedCallback::camera(void)
{
   return camera_;
}

void QuadrupedCallback::setMotorSpeeds(void)
{
/// right-side
   float frh_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("base"),
      *body_name_to_body_.at("fr_hip"),
      1,
      2,
      1.f
   );

   motor_name_to_motor_["fr_hip"]->angularSpeed = 5.f * (hip_rotation_angle_setpoint_ - frh_theta);

   float frt_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("fr_hip"),
      *body_name_to_body_.at("fr_thigh"),
      2,
      0,
      -1.f
   );

   motor_name_to_motor_["fr_thigh"]->angularSpeed = 5.f * (hip_flexion_angle_setpoint_ - frt_theta);

   float frc_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("fr_thigh"),
      *body_name_to_body_.at("fr_calf"),
      2,
      0,
      -1.f
   );

   motor_name_to_motor_["fr_calf"]->angularSpeed = 5.f * (knee_flexion_angle_setpoint_ - frc_theta);

   float brh_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("base"),
      *body_name_to_body_.at("br_hip"),
      1,
      2,
      1.f
   );

   motor_name_to_motor_["br_hip"]->angularSpeed = 5.f * (hip_rotation_angle_setpoint_ - brh_theta);

   float brt_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("br_hip"),
      *body_name_to_body_.at("br_thigh"),
      2,
      0,
      -1.f
   );

   motor_name_to_motor_["br_thigh"]->angularSpeed = 5.f * (hip_flexion_angle_setpoint_ - brt_theta);

   float brc_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("br_thigh"),
      *body_name_to_body_.at("br_calf"),
      2,
      0,
      -1.f
   );

   motor_name_to_motor_["br_calf"]->angularSpeed = 5.f * (knee_flexion_angle_setpoint_ - brc_theta);

/// left-side
   float flh_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("base"),
      *body_name_to_body_.at("fl_hip"),
      1,
      2,
      1.f
   );

   motor_name_to_motor_["fl_hip"]->angularSpeed = 5.f * (hip_rotation_angle_setpoint_ - flh_theta);

   float flt_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("fl_hip"),
      *body_name_to_body_.at("fl_thigh"),
      2,
      0,
      -1.f
   );

   motor_name_to_motor_["fl_thigh"]->angularSpeed = 5.f * (hip_flexion_angle_setpoint_ - flt_theta);

   float flc_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("fl_thigh"),
      *body_name_to_body_.at("fl_calf"),
      2,
      0,
      -1.f
   );

   motor_name_to_motor_["fl_calf"]->angularSpeed = 5.f * (knee_flexion_angle_setpoint_ - flc_theta);

   float blh_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("base"),
      *body_name_to_body_.at("bl_hip"),
      1,
      2,
      1.f
   );

   motor_name_to_motor_["bl_hip"]->angularSpeed = 5.f * (hip_rotation_angle_setpoint_ - blh_theta);

   float blt_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("bl_hip"),
      *body_name_to_body_.at("bl_thigh"),
      2,
      0,
      -1.f
   );

   motor_name_to_motor_["bl_thigh"]->angularSpeed = 5.f * (hip_flexion_angle_setpoint_ - blt_theta);

   float blc_theta = calculateRevoluteMotorTheta(
      *body_name_to_body_.at("bl_thigh"),
      *body_name_to_body_.at("bl_calf"),
      2,
      0,
      -1.f
   );

   motor_name_to_motor_["bl_calf"]->angularSpeed = 5.f * (knee_flexion_angle_setpoint_ - blc_theta);

}
