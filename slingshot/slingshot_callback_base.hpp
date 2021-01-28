#ifndef CALLBACKBASEHEADER
#define CALLBACKBASEHEADER

#include "handle.hpp"

#ifdef BUILD_VIZ
#include "camera.hpp"
#include "gui_callback_base.hpp"
#include "hid_interface.hpp"
#include "viz_renderer.hpp"
#endif

namespace slingshot
{

   class CallbackBase
   {
      public:
         CallbackBase(void)
            : default_timeout_(30.f)
         { }

         CallbackBase(float sim_timeout_time)
            : default_timeout_(sim_timeout_time)
         { }

         virtual ~CallbackBase(void)
         { }

         // This optional method is called once after the physics engine is
         // configured with a scenario.
         virtual void post_setup(oy::Handle & handle)
         {
            (void)handle;
         }

         // Return true to step the simulation. Return false to not step the
         // simulation.
         virtual bool operator()(oy::Handle & handle) = 0;

         // Return true to terminate the simulation.
         virtual bool terminate(float elapsed_sim_time)
         {
            return (elapsed_sim_time >= default_timeout_);
         }

#ifdef BUILD_VIZ

         virtual void parse_gui(
            viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
         )
         {
            (void)renderer;
            (void)fzx_to_viz_ids;
         }

         // Provides a custom ImGui interface to the renderer.
         virtual viz::GuiCallbackBase * gui(void)
         {
            return nullptr;
         }

         // Provide a custom camera for the renderer.
         virtual viz::Camera & camera(void) = 0;

         // Provide a custom camera controller/HID interface for the renderer.
         virtual viz::HIDInterface & hid(void) = 0;
#endif

      private:

         float default_timeout_;

   };

}

#endif
