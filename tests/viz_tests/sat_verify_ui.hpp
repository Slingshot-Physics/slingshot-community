#ifndef SAT_UI_HEADER
#define SAT_UI_HEADER

#include "gui_callback_base.hpp"

#include "sat.hpp"

#include "attitudeutils.hpp"
#include "slingshot_types.hpp"
#include "geometry_types.hpp"
#include "geometry_type_converters.hpp"
#include "support_functions.hpp"
#include "viz_types.hpp"

#include <cmath>
#include <vector>

#include "shape_loader_ui.hpp"

class SatVerify : public viz::GuiCallbackBase
{
   public:
      SatVerify(void)
         : shape_loaders{
            {"shape A", {0.f, 0.f, 2.f}},
            {"shape B", {0.f, 0.f, -2.f}}
         }
         , ui_modified_(false)
         , sphere_(
            geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::SPHERE)
         )
      {
         update_render_meshes();

         for (unsigned int i = 0; i < sphere_.numVerts; ++i)
         {
            geometry::types::transform_t temp_trans;
            temp_trans.translate = 1.5f * sphere_.verts[i];
            temp_trans.rotate = makeVectorUp(sphere_.verts[i]).transpose();
            temp_trans.scale = identityMatrix();
            temp_trans.scale(0, 0) = 0.1f;
            temp_trans.scale(1, 1) = 0.1f;
            cylinder_transforms.push_back(temp_trans);
            cylinder_colors.push_back({1.f, 0.f, 0.f, 1.f});
         }
      }

      void operator()(void)
      {
         ImGui::Begin("SAT Collision UI");

         no_window();

         ImGui::End();
      }

      bool no_window(void);

      bool ui_modified(void) const
      {
         return ui_modified_;
      }

      const geometry::types::satResult_t sat_out(void) const
      {
         return sat_out_;
      }

      template <typename ShapeA_T, typename ShapeB_T>
      void evaluateSat(const ShapeA_T shape_a, const ShapeB_T shape_b)
      {
         geometry::ShapeSupport<ShapeA_T> support_a(
            shape_loaders[0].trans_B_to_W(), shape_a
         );
         geometry::ShapeSupport<ShapeB_T> support_b(
            shape_loaders[1].trans_B_to_W(), shape_b
         );

         std::vector<float> heights;
         float max_height = -1.f * __FLT_MAX__;
         float min_height = __FLT_MAX__;
         float avg_height = 0.f;

         for (unsigned int i = 0; i < sphere_.numVerts; ++i)
         {
            float sigma = sphere_.verts[i].dot(
               support_a.boundSupportWorldDir(sphere_.verts[i]).vert - support_b.boundSupportWorldDir(-1.f * sphere_.verts[i]).vert
            );
            heights.push_back(sigma);

            if (sigma > max_height)
            {
               max_height = sigma;
            }
            if (sigma < min_height)
            {
               min_height = sigma;
            }

            avg_height += sigma / sphere_.numVerts;
         }

         const float y_max = 3.f;
         const float y_min = 1.f;

         float m = (y_max - y_min) / (max_height - min_height);

         for (unsigned int i = 0; i < sphere_.numVerts; ++i)
         {
            float scaled_sigma = m * (heights[i] - min_height) + y_min;
            cylinder_transforms[i].scale(2, 2) = scaled_sigma;
            // float red = (scaled_sigma - y_min) / y_max;
            float red = (std::tanh(heights[i] - avg_height) + 1.f) / 2.f;
            float blue = 1.f - red;
            cylinder_colors[i][0] = red;
            cylinder_colors[i][2] = blue;
         }
      }

      ShapeLoader shape_loaders[2];

      data_triangleMesh_t shape_render_data[2];

      std::vector<geometry::types::transform_t> cylinder_transforms;

      std::vector<viz::types::basic_color_t> cylinder_colors;

   private:

      bool ui_modified_;

      geometry::types::satResult_t sat_out_;

      const geometry::types::triangleMesh_t sphere_;

      void update_render_meshes(void)
      {
         geometry::types::triangleMesh_t temp_mesh;
         for (int i = 0; i < 2; ++i)
         {
            temp_mesh = shape_loaders[i].mesh();
            geometry::converters::to_pod(temp_mesh, &(shape_render_data[i]));
         }
      }

      void collision_multiplexor(void)
      {
         geometry::types::shape_t shape_a = shape_loaders[0].shape();
         geometry::types::shape_t shape_b = shape_loaders[1].shape();

         geometry::types::transform_t trans_A_to_W = shape_loaders[0].trans_B_to_W();
         geometry::types::transform_t trans_B_to_W = shape_loaders[1].trans_B_to_W();

         if (
            (shape_a.shapeType == geometry::types::enumShape_t::CUBE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CUBE)
         )
         {
            evaluateSat(shape_a.cube, shape_b.cube);
         }
         else if (
            (shape_a.shapeType == geometry::types::enumShape_t::CUBE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::SPHERE)
         )
         {
            evaluateSat(shape_a.cube, shape_b.sphere);
         }
         else if (
            (shape_a.shapeType == geometry::types::enumShape_t::CUBE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CAPSULE)
         )
         {
            evaluateSat(shape_a.cube, shape_b.capsule);
         }
         else if (
            (shape_a.shapeType == geometry::types::enumShape_t::CUBE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CYLINDER)
         )
         {
            evaluateSat(shape_a.cube, shape_b.cylinder);
         }
         else if(
            (shape_a.shapeType == geometry::types::enumShape_t::SPHERE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::SPHERE)
         )
         {
            evaluateSat(shape_a.sphere, shape_b.sphere);
         }
         else if(
            (shape_a.shapeType == geometry::types::enumShape_t::SPHERE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CAPSULE)
         )
         {
            evaluateSat(shape_a.sphere, shape_b.capsule);
         }
         else if(
            (shape_a.shapeType == geometry::types::enumShape_t::SPHERE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CYLINDER)
         )
         {
            evaluateSat(shape_a.sphere, shape_b.cylinder);
         }
         else if (
            (shape_a.shapeType == geometry::types::enumShape_t::CAPSULE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CAPSULE)
         )
         {
            evaluateSat(shape_a.capsule, shape_b.capsule);
         }
         else if (
            (shape_a.shapeType == geometry::types::enumShape_t::CAPSULE) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CYLINDER)
         )
         {
            evaluateSat(shape_a.capsule, shape_b.cylinder);
         }
         else if (
            (shape_a.shapeType == geometry::types::enumShape_t::CYLINDER) &&
            (shape_b.shapeType == geometry::types::enumShape_t::CYLINDER)
         )
         {
            evaluateSat(shape_a.cylinder, shape_b.cylinder);
         }
         else
         {
            sat_out_.collision = false;
            std::cout << "unsupported collision between " << static_cast<int>(shape_a.shapeType) << ", " << static_cast<int>(shape_b.shapeType) << "\n";
         }
      }
};

#endif
