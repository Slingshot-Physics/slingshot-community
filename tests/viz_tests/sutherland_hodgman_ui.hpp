#include "geometry.hpp"
#include "gui_callback_base.hpp"
#include "polygon_ui.hpp"
#include "vector3.hpp"

#include <algorithm>
#include <vector>

void convert_vec_of_vecs_to_polygon(
   const std::vector<Vector3> & vec_poly, geometry::types::polygon50_t & poly
)
{
   poly.numVerts = std::min((size_t )50, vec_poly.size());
   for (unsigned int i = 0; i < poly.numVerts; ++i)
   {
      poly.verts[i] = vec_poly[i];
   }
}

class MeddlingGui : public viz::GuiCallbackBase
{
   public:
      MeddlingGui(void)
         : clip_poly("clip", 25)
         , subject_poly("subject", 25)
      { }

      void operator()(void);

      bool no_window(void);

      PolygonUI clip_poly;

      PolygonUI subject_poly;

      std::vector<Vector3> intersection;

   private:

      void clip_polygons(void);

};
