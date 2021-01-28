#ifndef GRAHAM_SCAN_UI_HEADER
#define GRAHAM_SCAN_UI_HEADER

#include "polygon.hpp"
#include "geometry_types.hpp"
#include "polygon_ui.hpp"

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
         : polygon("poly", 25)
         , num_saves_(0)
      {
         hull.numVerts = 0;
      }

      void operator()(void);

      bool no_window(void);

      PolygonUI polygon;

      geometry::types::polygon50_t hull;

   private:
      unsigned int num_saves_;

      void make_hull(void);
};

#endif
