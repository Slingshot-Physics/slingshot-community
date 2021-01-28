#ifndef DEFAULT_GUI_HEADER
#define DEFAULT_GUI_HEADER

#include "gui_callback_base.hpp"
#include "vector3.hpp"

struct QuadrupedGuiState_t
{
   QuadrupedGuiState_t(void)
      : run_sim(false)
      , step_one(false)
      , show_grid(true)
      , camera_speed(0.5f)
      , num_meshes(0)
      , num_draw_calls(0)
      , camera_direction(0.f, 1.f, 0.f)
      , light_direction(6.f, 12.f, 5.f)
      , hip_rotation_angle_setpoint(0.f)
      , hip_flexion_angle_setpoint(0.f)
      , knee_flexion_angle_setpoint(0.f)
   { }

   bool run_sim;
   bool step_one;
   bool show_grid;
   float camera_speed;
   unsigned int num_meshes;
   unsigned int num_draw_calls;
   Vector3 camera_direction;
   Vector3 light_direction;
   float hip_rotation_angle_setpoint;
   float hip_flexion_angle_setpoint;
   float knee_flexion_angle_setpoint;
};

class QuadrupedGuiCallback : public viz::GuiCallbackBase
{
   public:

      QuadrupedGuiCallback(void)
         : clear_color_(0.45f, 0.55f, 0.60f, 1.00f)
      { }

      void operator()(void);

      bool no_window(void);

      QuadrupedGuiState_t & getState(void)
      {
         return state_;
      }

   private:
      ImVec4 clear_color_;

      QuadrupedGuiState_t state_;
};

#endif
