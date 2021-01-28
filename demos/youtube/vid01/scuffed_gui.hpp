#ifndef SCUFFED_LEAGUE_GUI_HEADER
#define SCUFFED_LEAGUE_GUI_HEADER

#include "gui_callback_base.hpp"
#include "vector3.hpp"

struct ScuffedGuiState_t
{
   float score;
};

class ScuffedGui : public viz::GuiCallbackBase
{
   public:

      ScuffedGui(void)
         : state_{0.f}
      { }

      void operator()(void);

      bool no_window(void);

      ScuffedGuiState_t & getState(void)
      {
         return state_;
      }

   private:

      ScuffedGuiState_t state_;
};

#endif
