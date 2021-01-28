#ifndef CAMERA_CONTROL_GUI_HEADER
#define CAMERA_CONTROL_GUI_HEADER

#include "gui_callback_base.hpp"

#include "slingshot_types.hpp"

struct CameraGuiState_t
{
   bool show_grid;
   float camera_speed;
   unsigned int num_meshes;
   unsigned int num_draw_calls;
   Vector3 camera_direction;
};

class CameraGui : public viz::GuiCallbackBase
{
   public:

      void operator()(void)
      { }

      bool no_window(void);

      CameraGuiState_t & getState(void)
      {
         return state_;
      }

   private:

      CameraGuiState_t state_;

};

#endif
