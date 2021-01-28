#ifndef POLYHEDRON_LOADER_UI_HEADER
#define POLYHEDRON_LOADER_UI_HEADER

#include "geometry.hpp"
#include "gui_callback_base.hpp"
#include "matrix33.hpp"
#include "vector3.hpp"

#include "transform_ui.hpp"

#include <algorithm>
#include <map>
#include <sstream>
#include <string>

class PolyhedronLoader : public viz::GuiCallbackBase
{
   static int counter_;

   public:

      PolyhedronLoader(const std::string & name, Vector3 pos)
         : transform_ui_("transform")
         , polyhedron_id_(geometry::types::enumShape_t::CUBE)
         , shape_changed_(true)
      {
         poly_types_and_names_[geometry::types::enumShape_t::CUBE] = std::string("cube");
         poly_types_and_names_[geometry::types::enumShape_t::CYLINDER] = std::string("cylinder");
         poly_types_and_names_[geometry::types::enumShape_t::SPHERE] = std::string("sphere");
         poly_types_and_names_[geometry::types::enumShape_t::CAPSULE] = std::string("capsule");
         transform_ui_.trans_B_to_W().translate = pos;

         load_mesh();

         offset_ = counter_;
         ++counter_;

         std::ostringstream ugh;
         ugh << name << " " << offset_;
         name_ = ugh.str();
      }

      void operator()(void);

      bool no_window(void);

      bool polyhedron_changed(void) const
      {
         return shape_changed_;
      }

      geometry::types::transform_t & trans_B_to_W(void)
      {
         return transform_ui_.trans_B_to_W();
      }

      geometry::types::triangleMesh_t & mesh(void)
      {
         return mesh_;
      }

      geometry::types::enumShape_t shape_type(void)
      {
         return static_cast<geometry::types::enumShape_t>(polyhedron_id_);
      }

      geometry::types::convexPolyhedron_t & polyhedron(void)
      {
         return polyhedron_;
      }

      geometry::types::gaussMapMesh_t & gauss_map(void)
      {
         return gauss_map_;
      }

      void load(
         const geometry::types::transform_t & trans_B_to_W,
         geometry::types::enumShape_t shape_type
      )
      {
         transform_ui_.trans_B_to_W() = trans_B_to_W;

         geometry::types::enumShape_t new_polyhedron_id = shape_type;
         shape_changed_ = new_polyhedron_id != polyhedron_id_;
         polyhedron_id_ = new_polyhedron_id;

         load_mesh();
      }

   private:

      std::string name_;

      std::map<geometry::types::enumShape_t, std::string> poly_types_and_names_;

      TransformUi transform_ui_;

      bool shape_changed_;

      int offset_;

      geometry::types::enumShape_t polyhedron_id_;

      geometry::types::triangleMesh_t mesh_;

      geometry::types::convexPolyhedron_t polyhedron_;

      geometry::types::gaussMapMesh_t gauss_map_;

      void load_mesh(void);

};

#endif
