#ifndef SLINGSHOT_HEADER
#define SLINGSHOT_HEADER

#ifdef BUILD_VIZ
#include "viz_renderer.hpp"
#endif

#include "slingshot_callback_base.hpp"
#include "data_model_io.h"
#include "handle.hpp"

#include <chrono>
#include <map>
#include <string>

namespace slingshot
{
   // This class is a wrapper around fzx and viz. It allows the user to supply
   // a functor with relevant fzx API calls, eliminating unnecessary
   // boilerplate on the user side. This class is the first attempt at an API
   // that explicitly avoids multithreading.
   class api
   {
      public:

#ifdef BUILD_VIZ
         api(
            const data_scenario_t & scenario,
            const data_vizConfig_t & viz_config,
            CallbackBase * user_callback
         );

         api(
            const data_scenario_t & scenario,
            const data_vizConfig_t & viz_config,
            const std::string & window_name,
            CallbackBase * user_callback
         );

         api(
            const data_vizScenarioConfig_t & viz_scenario_config,
            CallbackBase * user_callback
         );

         api(
            const data_vizScenarioConfig_t & viz_scenario_config,
            const std::string & window_name,
            CallbackBase * user_callback
         );
#endif

         // Constructor that runs the API without viz. Requires the user to
         // supply a maximum run time.
         api(
            const data_scenario_t & scenario,
            CallbackBase * user_callback
         );

         ~api(void);

         // Blocking call that updates the meshes in the renderer and calls the
         // user-defined callback to step the simulation forward. Terminates if
         // the renderer window is closed.
         void loop(void);

      private:

         void initialize_renderer(void);

         void update_renderer(void);

         void update_meshes(void);

         oy::Handle handle_;

         // User-defined (or default) callback that applies forces to any of
         // the bodies in the scenario.
         CallbackBase * callback_;

#ifdef BUILD_VIZ
         // Pointer to a renderer. This is initialized with a viz config file.
         viz::VizRenderer * renderer_;
#endif

         // viz configuration. Kept around for posterity.
         data_vizConfig_t viz_config_;

         // Bool indicating if the default physics callback was used. If true,
         // the destructor frees the heap-allocated default callback.
         bool use_default_callback_;

         std::chrono::high_resolution_clock::time_point frame_start_time_;

         std::chrono::high_resolution_clock::time_point sim_start_time_;

         bool stop_sim_;

         bool realtime_;

         std::map<int, trecs::uid_t> scen_id_to_sim_id_;

         // Mapping of body IDs in the engine to mesh IDs in viz. Naively
         // assumes that meshes will not be deleted or added during execution.
         std::map<trecs::uid_t, int> fzx_to_viz_ids_;

         // Uses a mixture of system calls to sleep and spin locks to make the
         // main thread wait for very close to an exact duration.
         void exact_sleep_us(unsigned int sleep_time_us);

   };
}

#endif
