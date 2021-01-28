#ifndef LINE_LOADER_UI_HEADER
#define LINE_LOADER_UI_HEADER

#include "gui_callback_base.hpp"
#include "vector3.hpp"

class LineLoader : public viz::GuiCallbackBase
{
   public:

      LineLoader(const char * prefix)
         : prefix_(prefix)
         , line_type(0)
         , line_type_names_{"line", "ray", "segment"}
         , combo_name_("line type")
         , start_point_name_("start point")
         , slope_name_("slope")
         , length_name_("length")
      {
         prepend_prefix(combo_name_);
         prepend_prefix(start_point_name_);
         prepend_prefix(slope_name_);
         prepend_prefix(length_name_);
      }

      void operator()(void);

      bool no_window(void);

      Vector3 line_points[2];

      Vector3 start;

      Vector3 slope;

      float length;

      // 0 - line
      // 1 - ray
      // 2 - segment
      int line_type;

   private:

      void update_line_points(void);

      void prepend_prefix(std::string & name);

      std::string prefix_;

      std::string combo_name_;

      std::string start_point_name_;

      std::string slope_name_;

      std::string length_name_;

      std::string line_type_names_[3];

};

#endif
