#ifndef TRANSFORM_UI_HEADER
#define TRANSFORM_UI_HEADER

#include "geometry_types.hpp"
#include "gui_callback_base.hpp"
#include "matrix33.hpp"
#include "vector3.hpp"

#include <sstream>
#include <string>

class TransformUi : public viz::GuiCallbackBase
{
   static int counter_;

   public:
      TransformUi(const std::string & name, bool force_diagonal_scale=false)
         : force_diagonal_scale_(force_diagonal_scale)
         , trans_B_to_W_{identityMatrix(), identityMatrix(), {0.f, 0.f, 0.f}}
      {
         offset_ = counter_;
         ++counter_;

         std::ostringstream ugh;
         ugh << name << " " << offset_;
         name_ = ugh.str();
      }

      void operator()(void)
      {
         ImGui::Begin("Transform UI");

         no_window();

         ImGui::End();
      }

      bool no_window(void);

      geometry::types::transform_t & trans_B_to_W(void)
      {
         return trans_B_to_W_;
      }

   private:

      bool force_diagonal_scale_;

      geometry::types::transform_t trans_B_to_W_;

      std::string name_;

      int offset_;
};

#endif
