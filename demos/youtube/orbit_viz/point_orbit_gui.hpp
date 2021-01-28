#ifndef POINT_ORBIT_GUI_HEADER
#define POINT_ORBIT_GUI_HEADER

#include "gui_callback_base.hpp"
#include "vector3.hpp"

#include <cmath>

struct PointOrbitGuiState_t
{
   bool run_sim;
   bool show_grid;
   Vector3 look_point;
   float orbit_radius;
   float orbit_frequency;
   float orbit_height;
   Vector3 light_direction;
};

class PointOrbitGui : public viz::GuiCallbackBase
{
   public:
      PointOrbitGui(void)
      {
         state_.run_sim = false;
         state_.show_grid = true;
         state_.look_point.Initialize(0.f, 0.f, 0.f);
         state_.orbit_radius = 5.f;
         state_.orbit_frequency = 0.f;
         state_.orbit_height = 1.f;
         state_.light_direction.Initialize(6.f, 12.f, 5.f);

         light_theta_deg_ = atan2f(
            state_.light_direction[1],
            state_.light_direction[0]
         ) * rad2deg;

         light_psi_deg_ = atan2f(
            state_.light_direction[2],
            sqrtf(
               state_.light_direction[0] * state_.light_direction[0] + \
               state_.light_direction[1] * state_.light_direction[1]
            )
         ) * rad2deg;

      }

      void operator()(void);

      bool no_window(void);

      PointOrbitGuiState_t & getState(void)
      {
         return state_;
      }

   private:
      PointOrbitGuiState_t state_;

      float light_theta_deg_;

      float light_psi_deg_;

      const float rad2deg = 180.f / M_PI;
      const float deg2rad = M_PI / 180.f;

};

#endif
