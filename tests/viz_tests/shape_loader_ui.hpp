#ifndef SHAPE_LOADER_UI_HEADER
#define SHAPE_LOADER_UI_HEADER

#include "geometry.hpp"
#include "gui_callback_base.hpp"
#include "matrix33.hpp"
#include "vector3.hpp"

#include "transform_ui.hpp"

#include <sstream>
#include <string>
#include <map>

class ShapeLoader : public viz::GuiCallbackBase
{
   static int counter_;

   public:
      ShapeLoader(
         const std::string & name,
         const Vector3 & pos,
         bool force_diagonal_scale=false
      )
         : selected_shape_type_id_(shape_to_int(geometry::types::enumShape_t::SPHERE))
         , shape_{
            geometry::types::enumShape_t::SPHERE, {1.f}
         }
         , shape_enums_to_names_{
            {geometry::types::enumShape_t::CAPSULE, "capsule"},
            {geometry::types::enumShape_t::CYLINDER, "cylinder"},
            {geometry::types::enumShape_t::SPHERE, "sphere"},
            {geometry::types::enumShape_t::CUBE, "cube"}
         }
         , transform_ui_("transform", force_diagonal_scale)
         , mesh_{
            geometry::mesh::loadDefaultShapeMesh(shape_.shapeType)
         }
      {
         mesh_ = geometry::mesh::loadShapeMesh(shape_);
         transform_ui_.trans_B_to_W().translate = pos;
         transform_ui_.trans_B_to_W().rotate = identityMatrix();
         transform_ui_.trans_B_to_W().scale = identityMatrix();
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
         return transform_ui_.trans_B_to_W();
      }

      geometry::types::shape_t & shape(void)
      {
         return shape_;
      }

      const geometry::types::triangleMesh_t & mesh(void) const
      {
         return mesh_;
      }

      void load(
         const geometry::types::transform_t & trans_B_to_W,
         geometry::types::shape_t shape
      )
      {
         transform_ui_.trans_B_to_W() = trans_B_to_W;
         shape_ = shape;
         mesh_ = geometry::mesh::loadShapeMesh(shape_);
         selected_shape_type_id_ = shape_to_int(shape_.shapeType);
      }

   private:

      std::string name_;

      int offset_;

      int selected_shape_type_id_;

      geometry::types::shape_t shape_;

      std::map<geometry::types::enumShape_t, std::string> shape_enums_to_names_;

      TransformUi transform_ui_;

      // The untransformed mesh for collision algorithms
      geometry::types::triangleMesh_t mesh_;

      int shape_to_int(geometry::types::enumShape_t shape_type)
      {
         return static_cast<int>(shape_type);
      }

      geometry::types::enumShape_t int_to_shape(int shape_val)
      {
         return static_cast<geometry::types::enumShape_t>(shape_val);
      }

      bool capsule_widget(void);

      bool cylinder_widget(void);

      bool sphere_widget(void);

      bool cube_widget(void);
};


#endif
