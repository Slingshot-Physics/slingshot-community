#include "default_callback.hpp"

#include "transform.hpp"

namespace slingshot
{
   DefaultCallback::DefaultCallback(void)
      : run_sim_(true)
      , step_one_(false)
      , rmb_hold_(false)
      , rmb_release_(true)
      , rmb_change_(false)
      , grabbed_body_uid_(-1)
      , spring_grabber_uid_(-1)
      , damper_grabber_uid_(-1)
      , grab_dist_(0.f)
#ifdef BUILD_VIZ
      , camera_controller_(camera_)
#endif
   {
#ifdef BUILD_VIZ
      camera_controller_.setLeftAltAsRmb(true);
#endif
   }

   bool DefaultCallback::operator()(oy::Handle & handle)
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
            temp_spring.springCoeff = -600.f;

            spring_grabber_uid_ = handle.addSpringForce(result.bodyId, -1, temp_spring);

            oy::types::forceVelocityDamper_t temp_damper;
            temp_damper.damperCoeff = -40.f;
            temp_damper.parentLinkPoint = temp_spring.parentLinkPoint;
            temp_damper.childLinkPoint = temp_spring.childLinkPoint;

            damper_grabber_uid_ = handle.addVelocityDamperForce(result.bodyId, -1, temp_damper);

            grabbed_body_uid_ = static_cast<trecs::uid_t>(result.bodyId);

            std::cout << "grabbed body UID " << grabbed_body_uid_ << "\n";
            auto scen_to_sim_ids = handle.getIdMapping();
            for (const auto & scen_sim_id : scen_to_sim_ids)
            {
               if (scen_sim_id.second == grabbed_body_uid_)
               {
                  std::cout << "\twith scenario id " << scen_sim_id.first << "\n";
               }
            }
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

      // Roll it!
      return true;
   }

#ifdef BUILD_VIZ
   void DefaultCallback::parse_gui(
      viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
   )
   {
      (void)fzx_to_viz_ids;
      DefaultGuiState_t & gui_state = gui_.getState();

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
   }

   viz::GuiCallbackBase * DefaultCallback::gui(void)
   {
      return &gui_;
   }

   viz::HIDInterface & DefaultCallback::hid(void)
   {
      viz::HIDInterface & hid_controller = camera_controller_;
      return hid_controller;
   }

   viz::Camera & DefaultCallback::camera(void)
   {
      return camera_;
   }
#endif

}
