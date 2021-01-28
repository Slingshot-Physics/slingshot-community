#ifndef DEFAULT_CALLBACK_HEADER
#define DEFAULT_CALLBACK_HEADER

#ifdef BUILD_VIZ
#include "default_camera_controller.hpp"
#include "default_gui.hpp"
#endif

#include "slingshot_callback_base.hpp"

namespace slingshot
{

   class DefaultCallback : public CallbackBase
   {
      public:
         DefaultCallback(void);

         ~DefaultCallback(void)
         { }

         bool operator()(oy::Handle & handle);

#ifdef BUILD_VIZ
         void parse_gui(
            viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
         );

         viz::GuiCallbackBase * gui(void);

         viz::HIDInterface & hid(void);

         viz::Camera & camera(void);
#endif

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

#ifdef BUILD_VIZ
         DefaultGuiCallback gui_;

         viz::Camera camera_;

         viz::DefaultCameraController camera_controller_;
#endif

   };

}

#endif
