#ifndef SIMPLE_CALLBACK_HEADER
#define SIMPLE_CALLBACK_HEADER

#include "slingshot_callback_base.hpp"

#include "point_orbit_camera_controller.hpp"
#include "point_orbit_gui.hpp"

class SimpleCallback : public slingshot::CallbackBase
{
   public:
      SimpleCallback(void)
         : camera_controller_(camera_, {0.f, 0.f, 0.f}, 1.f, 1.f, 1.f)
         , run_sim_(false)
         , rmb_hold_(false)
         , rmb_release_(true)
         , rmb_change_(false)
         , grabbed_body_uid_(-1)
         , spring_grabber_uid_(-1)
         , damper_grabber_uid_(-1)
         , grab_dist_(0.f)
      { }

      void post_setup(oy::Handle & handle);

      bool operator()(oy::Handle & handle);

      viz::HIDInterface & hid(void);

      viz::Camera & camera(void);

      void parse_gui(
         viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
      );

      viz::GuiCallbackBase * gui(void)
      {
         return &gui_;
      }

   private:

      PointOrbitGui gui_;

      viz::Camera camera_;

      viz::PointOrbitCameraController camera_controller_;

      bool run_sim_;

      bool rmb_hold_;

      bool rmb_release_;

      bool rmb_change_;

      trecs::uid_t grabbed_body_uid_;

      trecs::uid_t spring_grabber_uid_;

      trecs::uid_t damper_grabber_uid_;

      Vector3 camera_pos_;

      Vector3 camera_ray_slope_;

      float grab_dist_;

};

#endif
