#include "plinko_callback.hpp"

#include "random_utils.hpp"

#include "transform.hpp"

namespace slingshot
{
   PlinkoCallback::PlinkoCallback(void)
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
   {
      camera_controller_.setLeftAltAsRmb(true);
      camera_.setFarPlane(8.f);
   }

   bool PlinkoCallback::operator()(oy::Handle & handle)
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

      const auto body_uids = handle.getBodyUids();

      for (const auto body_uid : body_uids)
      {
         auto & body = handle.getBody(body_uid);
         if (!handle.stationary(body_uid) && body.linPos[2] <= -3.f)
         {
            body.linPos[0] = edbdmath::random_float(0.f, 0.4f);
            body.linPos[1] = edbdmath::random_float(-0.25f, 0.25f);
            body.linPos[2] = 3.75f + edbdmath::random_float(0.f, 0.25f);
            body.linVel[0] = 0.f;
            body.linVel[1] = 0.f;
            body.linVel[2] = edbdmath::random_float(-0.25f, 0.5f);
         }
      }

      // Roll it!
      return true;
   }

   void PlinkoCallback::parse_gui(
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

   viz::GuiCallbackBase * PlinkoCallback::gui(void)
   {
      return &gui_;
   }

   viz::HIDInterface & PlinkoCallback::hid(void)
   {
      viz::HIDInterface & hid_controller = camera_controller_;
      return hid_controller;
   }

   viz::Camera & PlinkoCallback::camera(void)
   {
      return camera_;
   }

}
