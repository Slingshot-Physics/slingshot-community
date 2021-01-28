#include "viztypeconverters.hpp"

#include "opengl_renderer.hpp"

#include "matrix33.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"

namespace viz
{

namespace converters
{
   // Convert from FZX coordinates to GL coordinates.
   //    x_fzx = x_gl
   //    y_fzx = -z_gl
   //    z_fzx = y_gl
   // Fucking column-major bullshit from the GLM library.
   static const glm::mat3 R_gl_from_fzx(
      1, 0, 0,
      0, 0, -1,
      0, 1, 0
   );

   static const Matrix33 R_fzx_from_gl(
      1, 0, 0,
      0, 0, 1,
      0, -1, 0
   );

   // Returns -1 if counter clockwise, returns 1 if clockwise. Only works for
   // convex polyhedra.
   int calcVertOrdering(
      const Vector3 & a,
      const Vector3 & b,
      const Vector3 & c,
      const Vector3 & center
   )
   {
      Vector3 tempNormal = (c - a).crossProduct(b - a);

      return tempNormal.dot(a - center) > 1e-7 ? 1 : -1;
   }

   void convert_gl_to_fzx(const glm::vec3 & vecIn, Vector3 & vecOut)
   {
      Vector3 glVecIn(vecIn[0], vecIn[1], vecIn[2]);
      vecOut = R_fzx_from_gl * glVecIn;
   }

   void convert_gl_to_fzx(const Vector3 & glVecIn, Vector3 & vecOut)
   {
      vecOut = R_fzx_from_gl * glVecIn;
   }

   void convert_fzx_to_gl(const Vector3 & vecIn, glm::vec3 & vecOut)
   {
      glm::vec3 glVecIn(vecIn[0], vecIn[1], vecIn[2]);
      vecOut = R_gl_from_fzx * glVecIn;
   }

   void convert_fzx_to_gl(const Vector3 & vecIn, Vector3 & glVecOut)
   {
      glVecOut = R_fzx_from_gl.transpose() * vecIn;
   }

   void convert_Vector3_to_GLMvec3(const Vector3 & vecIn, glm::vec3 & vecOut)
   {
      vecOut[0] = vecIn[0];
      vecOut[1] = vecIn[1];
      vecOut[2] = vecIn[2];
   }

   void convert_GLMvec3_to_Vector3(const glm::vec3 & vecIn, Vector3 & vecOut)
   {
      vecOut[0] = vecIn[0];
      vecOut[1] = vecIn[1];
      vecOut[2] = vecIn[2];
   }

   void convert_fzxRotM3_to_GLRotM4(
      const Matrix33 & mat_in, glm::mat4 & mat_out
   )
   {
      // Convert the edbdmath rotation matrix into a GLM rotation matrix in
      // ENU. The initialization looks like it's transposed because GLM
      // matrices are column-major.
      const Matrix33 & R = mat_in;
      glm::mat3 R_fzx(
         R(0, 0), R(1, 0), R(2, 0),
         R(0, 1), R(1, 1), R(2, 1),
         R(0, 2), R(1, 2), R(2, 2)
      );

      // Convert FZX coordinates into OpenGL coordinates (NUE).
      glm::mat3 R_glm = R_gl_from_fzx * R_fzx;

      // Copy the 3x3 matrix into the 4x4 matrix.
      glm::mat4 modelRot_glm(1.0f);
      for (unsigned int i = 0; i < 3; ++i)
      {
         for (unsigned int j = 0; j < 3; ++j)
         {
            modelRot_glm[i][j] = R_glm[i][j];
         }
      }

      mat_out = modelRot_glm;
   }

   void convert_fzxScaleM3_to_GLScaleM4(
      const Matrix33 & mat_in, glm::mat4 & mat_out
   )
   {
      const Matrix33 & S = mat_in;
      glm::mat3 S_fzx(
         S(0, 0), S(1, 0), S(2, 0),
         S(0, 1), S(1, 1), S(2, 1),
         S(0, 2), S(1, 2), S(2, 2)
      );

      // This commented out line *would be correct* if it was applied to a
      // vertex that's expressed in OpenGL coordinates. This function is
      // apologizing for the fact that all of the meshes sent to the GPU use
      // the z-is-up convention, instead of the OpenGL convention where
      // y-is-up.
      // glm::mat3 S_glm = S_fzx * glm::transpose(R_gl_from_fzx);
      glm::mat3 S_glm = S_fzx;

      // Copy the 3x3 matrix into the 4x4 matrix.
      glm::mat4 modelScale_glm(1.0f);
      for (unsigned int i = 0; i < 3; ++i)
      {
         for (unsigned int j = 0; j < 3; ++j)
         {
            modelScale_glm[i][j] = S_glm[i][j];
         }
      }

      mat_out = modelScale_glm;
   }

   // Converts a translation and rotation from Matrix33 and Vector3 types to
   // GLM Mat4 SE(4) type.
   void convert_fzxTrans_to_GLTransM4(
      const Matrix33 & rotate, const Vector3 & translate, glm::mat4 & mat_out
   )
   {
      // Rotate ENU position to GL (NUE) coordinates.
      glm::vec3 pos_glm;
      convert_Vector3_to_GLMvec3(translate, pos_glm);
      pos_glm = R_gl_from_fzx * pos_glm;

      // Convert rotation matrix to GL coordinates.
      glm::mat4 modelRot_glm(1.0f);
      convert_fzxRotM3_to_GLRotM4(rotate, modelRot_glm);

      // In order of matrix operations, rotate the model, then translate it.
      glm::mat4 modelTrans = glm::mat4(1.0f);
      mat_out = glm::translate(modelTrans, pos_glm) * modelRot_glm;
   }

   void convert_fzxTrans_to_GLTransM4(
      const Matrix33 & scale,
      const Matrix33 & rotate,
      const Vector3 & translate,
      glm::mat4 & mat_out
   )
   {
      // Rotate ENU position to GL (NUE) coordinates.
      glm::vec3 pos_fzx;
      convert_Vector3_to_GLMvec3(translate, pos_fzx);

      glm::vec3 pos_glm = R_gl_from_fzx * pos_fzx;
      glm::mat4 modelTrans_glm = glm::mat4(1.0f);
      modelTrans_glm = glm::translate(modelTrans_glm, pos_glm);

      // Convert rotation matrix to GL coordinates.
      glm::mat4 modelRot_glm(1.0f);
      convert_fzxRotM3_to_GLRotM4(rotate, modelRot_glm);

      glm::mat4 modelScale_glm(1.0f);
      convert_fzxScaleM3_to_GLScaleM4(scale, modelScale_glm);

      // In order of matrix operations: scale, rotate, then translate the
      // model.
      mat_out = modelTrans_glm * modelRot_glm * modelScale_glm;
   }

   void convert_data_to_basic_color(
      const data_vector4_t & vec4_in, viz::types::basic_color_t & vec4_out
   )
   {
      for (int i = 0; i < 4; ++i)
      {
         vec4_out[i] = vec4_in.v[i];
      }
   }

   void convert_data_to_meshProperties(
      const data_vizMeshProperties_t & props_in,
      viz::types::meshProperties_t & props_out
   )
   {
      convert_data_to_basic_color(props_in.color, props_out.color);
   }

   void convert_data_to_vizConfig(
      const data_vizConfig_t & viz_config_in,
      viz::types::config_t & viz_config_out
   )
   {
      viz_config_out.maxFps = viz_config_in.maxFps;
      viz_config_out.windowHeight = viz_config_in.windowHeight;
      viz_config_out.windowWidth = viz_config_in.windowWidth;

      for (int i = 0; i < 3; ++i)
      {
         viz_config_out.cameraPoint[i] = viz_config_in.cameraPoint.v[i];
         viz_config_out.cameraPos[i] = viz_config_in.cameraPos.v[i];
      }

      if ((viz_config_out.cameraPos - viz_config_out.cameraPoint).magnitudeSquared() < 1e-7)
      {
         viz_config_out.cameraPoint[3] = viz_config_out.cameraPos[3] - 1.f;
      }
   }

   viz::types::config_t convert_data_to_vizConfig(
      const data_vizConfig_t & viz_config_in
   )
   {
      viz::types::config_t result;
      convert_data_to_vizConfig(viz_config_in, result);
      return result;
   }

   std::vector<viz::types::basic_vertex_t> convert_Vector3_array_to_vertex_array(
      unsigned int numPoints,
      const Vector3 * points,
      const viz::types::basic_color_t & color
   )
   {
      std::vector<viz::types::basic_vertex_t> verts;
      verts.reserve(numPoints);
      for (unsigned int i = 0; i < numPoints; ++i)
      {
         viz::types::basic_vertex_t temp_vert;
         for (unsigned int j = 0; j < 3; ++j)
         {
            temp_vert.pos[j] = points[i][j];
         }

         temp_vert.color[0] = color[0];
         temp_vert.color[1] = color[1];
         temp_vert.color[2] = color[2];
         temp_vert.color[3] = color[3];

         verts.push_back(temp_vert);
      }

      return verts;
   }

   std::vector<viz::types::basic_vertex_t> convert_Vector3_array_to_vertex_array(
      const std::vector<Vector3> & points,
      const viz::types::basic_color_t & color
   )
   {
      std::vector<viz::types::basic_vertex_t> verts;
      verts.reserve(points.size());
      for (unsigned int i = 0; i < points.size(); ++i)
      {
         viz::types::basic_vertex_t temp_vert;
         for (unsigned int j = 0; j < 3; ++j)
         {
            temp_vert.pos[j] = points[i][j];
         }

         temp_vert.color[0] = color[0];
         temp_vert.color[1] = color[1];
         temp_vert.color[2] = color[2];
         temp_vert.color[3] = color[3];

         verts.push_back(temp_vert);
      }

      return verts;
   }

}

}
