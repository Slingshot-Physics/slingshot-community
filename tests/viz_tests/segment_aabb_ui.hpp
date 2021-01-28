#ifndef SEGMENT_AABB_UI_HEADER
#define SEGMENT_AABB_UI_HEADER

#include "aabb_ui.hpp"
#include "line_loader_ui.hpp"

class MeddlingGui : public viz::GuiCallbackBase
{
   public:
      MeddlingGui(void)
         : segment_loader("segment")
         , aabb_loader("aabb")
         , ui_modified_(false)
      { }

      void operator()(void);

      bool no_window(void);

      bool ui_modified(void) const
      {
         return ui_modified_;
      }

      LineLoader segment_loader;

      AABBUI aabb_loader;

      geometry::types::segmentClosestPoints_t closest_points;

   private:
      bool ui_modified_;

      void calculate_closest_points(void);
};

#endif
