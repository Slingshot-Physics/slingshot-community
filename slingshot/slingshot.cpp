#include "slingshot.hpp"

#include "default_callback.hpp"

#include <iostream>
#include <thread>

namespace slingshot
{
   api::api(
      const data_scenario_t & scenario,
      CallbackBase * user_callback
   )
      : handle_(scenario)
      , callback_(user_callback)
#ifdef BUILD_VIZ
      , renderer_(nullptr)
#endif      
      , use_default_callback_(user_callback == nullptr)
      , frame_start_time_(std::chrono::high_resolution_clock::now())
      , sim_start_time_(std::chrono::high_resolution_clock::now())
      , stop_sim_(false)
      , realtime_(false)
   {
      if (use_default_callback_)
      {
         callback_ = new DefaultCallback();
      }

      callback_->post_setup(handle_);

      scen_id_to_sim_id_ = handle_.getIdMapping();
   }

   api::~api(void)
   {
      if (use_default_callback_)
      {
         delete callback_;
         callback_ = nullptr;
      }

#ifdef BUILD_VIZ
      if (renderer_ != nullptr)
      {
         delete renderer_;
         renderer_ = nullptr;
      }
#endif
   }

   void api::loop(void)
   {
      while (true)
      {
         stop_sim_ = false;
#ifdef BUILD_VIZ
         std::chrono::high_resolution_clock::time_point start_time(
            std::chrono::high_resolution_clock::now()
         );

         // Run N physics updates, then render, and pause for the remaining
         // frame time (if any is available).
         if (renderer_ != nullptr && viz_config_.realtime)
         {
            unsigned int frame_time_ms = (1e3f / viz_config_.maxFps);
            unsigned int sim_updates_per_frame = (
               1.f / (handle_.dt() * viz_config_.maxFps)
            );

            for (unsigned int i = 0; i < sim_updates_per_frame; ++i)
            {
               if ((*callback_)(handle_))
               {
                  handle_.step();
               }
            }

            update_renderer();

            stop_sim_ |= (renderer_ == nullptr && callback_->terminate(handle_.getFrameCount() * handle_.dt()));

            std::chrono::duration<double, std::micro> exec_time_us = (
               std::chrono::high_resolution_clock::now() - start_time
            );

            if ((unsigned int )exec_time_us.count() < 1000 * frame_time_ms)
            {
               exact_sleep_us(1000 * frame_time_ms - (unsigned int )exec_time_us.count());
            }
         }
         // Run a physics update, only run the renderer if the max frames per
         // second are hit.
         else
         {
            if ((*callback_)(handle_))
            {
               handle_.step();
            }

            unsigned int frame_time_ms = (1e3f / viz_config_.maxFps);
            bool advance_frame = (
               (start_time - frame_start_time_).count() / 1e6f > frame_time_ms
            );
            if (advance_frame)
            {
               frame_start_time_ = std::chrono::high_resolution_clock::now();
               update_renderer();
            }
            stop_sim_ |= (renderer_ == nullptr && callback_->terminate(handle_.getFrameCount() * handle_.dt()));
         }
#else
         float elapsed_sim_time = handle_.getFrameCount() * handle_.dt();
         if ((*callback_)(handle_))
         {
            handle_.step();
         }

         stop_sim_ |= callback_->terminate(elapsed_sim_time);
#endif

         if (stop_sim_)
         {
            std::chrono::high_resolution_clock::time_point current_time(
               std::chrono::high_resolution_clock::now()
            );
            float elapsed_real_time = ((current_time - sim_start_time_).count() / 1e9);
            std::cout << "Terminating after " << elapsed_real_time << " wall clock seconds\n";
            std::cout << "Terminating after " << handle_.getFrameCount() * handle_.dt() << " sim seconds\n";
            break;
         }
      }
   }

   void api::exact_sleep_us(unsigned int sleep_time_us)
   {
      // Hard-coded minimum time to attempt to sleep for. You're at the
      // operating system's mercy when calling sleep, 3000us empirically seems
      // like the closest you can get to a minimum sleep-time resolution
      // without waiting longer than the desired sleep time.
      while (sleep_time_us > 3000)
      {
         std::chrono::high_resolution_clock::time_point start_time(
            std::chrono::high_resolution_clock::now()
         );
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
         std::chrono::duration<double, std::micro> one_step_sleep_time_us = (
            std::chrono::high_resolution_clock::now() - start_time
         );
         if (one_step_sleep_time_us.count() < sleep_time_us)
         {
            sleep_time_us -= one_step_sleep_time_us.count();
         }
         else
         {
            return;
         }
      }
   }

}
