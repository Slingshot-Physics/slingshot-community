#ifndef DEFAULT_GUI_HEADER
#define DEFAULT_GUI_HEADER

#include "gui_callback_base.hpp"
#include "vector3.hpp"

#include <vector>

struct DefaultGuiState_t
{
   bool run_sim;
   bool step_one;
   bool show_grid;
   float camera_speed;
   unsigned int num_meshes;
   unsigned int num_draw_calls;
   Vector3 camera_direction;
   Vector3 light_direction;
   std::vector<unsigned int> entity_uids;
};

class DefaultGuiCallback : public viz::GuiCallbackBase
{
   public:

      DefaultGuiCallback(void)
         : clear_color_(0.45f, 0.55f, 0.60f, 1.00f)
         , state_{
            false,
            false,
            true,
            0.5f,
            0,
            0,
            {0.f, 1.f, 0.f},
            {6.f, 12.f, 5.f},
            {}
         }
      { }

      void operator()(void);

      bool no_window(void);

      DefaultGuiState_t & getState(void)
      {
         return state_;
      }

   private:
      ImVec4 clear_color_;

      DefaultGuiState_t state_;
};

#endif
