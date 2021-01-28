#include "polyhedron_loader_ui.hpp"

#include "attitudeutils.hpp"
#include "data_model.h"
#include "gauss_map.hpp"
#include "geometry_type_converters.hpp"
#include "transform_utils.hpp"

#include <algorithm>

int PolyhedronLoader::counter_ = 0;

void PolyhedronLoader::operator()(void)
{
   ImGui::Begin("Polyhedron Loader");

   no_window();

   ImGui::End();
}

bool PolyhedronLoader::no_window(void)
{
   bool ui_modified = false;

   geometry::types::enumShape_t old_selection = polyhedron_id_;

   ImGui::SetNextItemOpen(true);
   if (ImGui::TreeNode(name_.c_str()))
   {
      if (
         ImGui::BeginCombo(
            "polyhedron",
            poly_types_and_names_[polyhedron_id_].c_str()
         )
      )
      {
         for (const auto & type_and_name : poly_types_and_names_)
         {
            bool is_selected = (polyhedron_id_ == type_and_name.first);
            if (ImGui::Selectable(type_and_name.second.c_str(), is_selected))
            {
               polyhedron_id_ = type_and_name.first;
            }

            if (is_selected)
            {
               ImGui::SetItemDefaultFocus();
            }
         }

         ImGui::EndCombo();
      }

      bool transform_changed = transform_ui_.no_window();

      shape_changed_ = old_selection != polyhedron_id_;
      ui_modified = (shape_changed_ || transform_changed);

      if (shape_changed_)
      {
         load_mesh();
      }

      ImGui::TreePop();
   }

   return ui_modified;
}

void PolyhedronLoader::load_mesh(void)
{
   mesh_ = geometry::mesh::loadDefaultShapeMesh(
      static_cast<geometry::types::enumShape_t>(polyhedron_id_)
   );

   test_utils::convert_triangleMesh_to_convexPolyhedron(
      mesh_, polyhedron_
   );

   gauss_map_ = geometry::mesh::loadGaussMap(
      static_cast<geometry::types::enumShape_t>(polyhedron_id_)
   );
}
