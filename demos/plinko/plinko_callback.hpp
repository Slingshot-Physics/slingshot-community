#ifndef PLINKO_CALLBACK_HEADER
#define PLINKO_CALLBACK_HEADER

#include "default_camera_controller.hpp"
#include "default_gui.hpp"
#include "slingshot_callback_base.hpp"

namespace slingshot
{
   class PlinkoCallback : public CallbackBase
   {
public:
         PlinkoCallback(void);

         ~PlinkoCallback(void)
         { }

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

         DefaultGuiCallback gui_;

         viz::Camera camera_;

         viz::DefaultCameraController camera_controller_;

   };
}

#endif
