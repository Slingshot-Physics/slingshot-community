#include "line_loader_ui.hpp"
#include "vector3.hpp"

class MeddlingGui : public viz::GuiCallbackBase
{
   public:
      MeddlingGui(void)
         : line_loader("segment")
         , segment_loader("line")
         , intersection(false)
         , touch_point(10000000.f, 0.f, 0.f)
      {}

      void operator()(void);

      bool no_window(void);

      LineLoader line_loader;

      LineLoader segment_loader;

      bool intersection;

      Vector3 touch_point;

   private:

      void calculate_intersection(void);
};
