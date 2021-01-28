#include "line_loader_ui.hpp"
#include "vector3.hpp"

struct segment_closest_points_t
{
   int num_points;
   Vector3 seg_a_points[2];
   Vector3 seg_b_points[2];
};

class MeddlingGui : public viz::GuiCallbackBase
{
   public:
      MeddlingGui(void)
         : segment_loader_a("segment a")
         , segment_loader_b("segment b")
         , ui_modified_(false)
      {
         closest_point_pairs.num_points = 0;
      }

      void operator()(void);

      bool no_window(void);

      bool ui_modified(void) const
      {
         return ui_modified_;
      }

      LineLoader segment_loader_a;

      LineLoader segment_loader_b;

      segment_closest_points_t closest_point_pairs;

   private:

      bool ui_modified_;

      void calculate_closest_points(void);
};
