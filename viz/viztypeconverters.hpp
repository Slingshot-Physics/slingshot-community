// The type converters for model transforms assume that meshes are uploaded
// with z-hat being up, which is not what OpenGL uses.

#ifndef VIZTYPECONVERTERS_HEADER
#define VIZTYPECONVERTERS_HEADER

#include "data_viz_config.h"
#include "data_viz_mesh_properties.h"
#include "viz_types.hpp"

// This is necessary for Mac - the latest version of GLM deprecates calls with
// radians.
#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

class Matrix33;
class Vector3;

namespace viz
{

namespace converters
{

   // Returns -1 if counter clockwise, returns 1 if clockwise. Only works for
   // convex polyhedra.
   int calcVertOrdering(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & center
   );

   void convert_gl_to_fzx(const glm::vec3 & vecIn, Vector3 & vecOut);

   void convert_gl_to_fzx(const Vector3 & glVecIn, Vector3 & vecOut);

   void convert_fzx_to_gl(const Vector3 & vecIn, glm::vec3 & vecOut);

   void convert_fzx_to_gl(const Vector3 & vecIn, Vector3 & glVecOut);

   void convert_Vector3_to_GLMvec3(const Vector3 & vecIn, glm::vec3 & vecOut);

   void convert_GLMvec3_to_Vector3(const glm::vec3 & vecIn, Vector3 & vecOut);

   // Converts a rotation matrix in FZX's coordinates (ENU, or z-up) to GL
   // coordinates (NUE, or y-up). This can only be used for rotation matrices.
   void convert_fzxRotM3_to_GLRotM4(
      const Matrix33 & mat_in, glm::mat4 & mat_out
   );

   // Converts a scale matrix in FZX's coordinates (ENU, or z-up) to GL
   // coordinates (NUE, or y-up). This can only be used for rotation matrices.
   void convert_fzxScaleM3_to_GLScaleM4(
      const Matrix33 & mat_in, glm::mat4 & mat_out
   );

   // Converts a rotation and translation from Matrix33 and Vector3 types to
   // GLM Mat4 SE(4) type in GL coordinates.
   void convert_fzxTrans_to_GLTransM4(
      const Matrix33 & rotate,
      const Vector3 & translate,
      glm::mat4 & mat_out
   );

   // Converts a scale, rotation, and translation from Matrix33 and Vector3
   // types to GLM Mat4 SE(4) type in GL coordinates.
   // This function assumes that the model transform will be applied to a
   // vector whose z-component points up (ENU), and that the resulting vector
   // will be in GL coordinates, where the y-component points up (NUE).
   void convert_fzxTrans_to_GLTransM4(
      const Matrix33 & scale,
      const Matrix33 & rotate,
      const Vector3 & translate,
      glm::mat4 & mat_out
   );

   void convert_data_to_basic_color(
      const data_vector4_t & vec4_in, viz::types::basic_color_t & vec4_out
   );

   void convert_data_to_meshProperties(
      const data_vizMeshProperties_t & props_in,
      viz::types::meshProperties_t & props_out
   );

   void convert_data_to_vizConfig(
      const data_vizConfig_t & viz_config_in,
      viz::types::config_t & viz_config_out
   );

   viz::types::config_t convert_data_to_vizConfig(
      const data_vizConfig_t & viz_config_in
   );

   std::vector<viz::types::basic_vertex_t>  convert_Vector3_array_to_vertex_array(
      unsigned int numPoints,
      const Vector3 * points,
      const viz::types::basic_color_t & color
   );

   std::vector<viz::types::basic_vertex_t> convert_Vector3_array_to_vertex_array(
      const std::vector<Vector3> & points,
      const viz::types::basic_color_t & color
   );

}

}

#endif
