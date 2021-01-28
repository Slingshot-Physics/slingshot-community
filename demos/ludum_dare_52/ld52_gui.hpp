#ifndef LD52_GUI_HEADER
#define LD52_GUI_HEADER

#include "gui_callback_base.hpp"
#include "vector3.hpp"

struct LD52GuiState_t
{
   bool show_instructions;
   float score;
   unsigned int lives_remaining;
   bool win;
   bool end_game;
};

class LD52Gui : public viz::GuiCallbackBase
{
   public:

      LD52Gui(unsigned int lives_remaining)
         : state_{
            true, 0.f, lives_remaining, false, false
         }
      { }

      void operator()(void);

      bool no_window(void);

      void instructions(void);

      void game_over(void);

      void win(void);

      LD52GuiState_t & getState(void)
      {
         return state_;
      }

   private:

      LD52GuiState_t state_;
};

#endif
