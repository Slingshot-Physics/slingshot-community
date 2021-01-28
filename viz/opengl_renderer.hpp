// OpenGLRenderer keeps track of all renderable objects, updates render
// buffers, and selects which shaders to use.

#ifndef OPENGL_RENDERER_HEADER
#define OPENGL_RENDERER_HEADER

#include "basic_shader.hpp"
#include "data_triangle_mesh.h"
#include "flat_mesh_buffer.hpp"
#include "grid.hpp"
#include "viztypeconverters.hpp"

#include <map>
#include <utility>
#include <vector>

#define FUCKOFFBUFFERSIZE (512 * 1024)

#ifndef VERTEX_MVP_SHADER_LOC
#define VERTEX_MVP_SHADER_LOC ""
#endif

#ifndef LINE_FRAGMENT_SHADER_LOC
#define LINE_FRAGMENT_SHADER_LOC ""
#endif

#ifndef INSTANCED_VERTEX_SHADER_LOC
#define INSTANCED_VERTEX_SHADER_LOC ""
#endif

#ifndef EMITTER_FRAGMENT_SHADER_LOC
#define EMITTER_FRAGMENT_SHADER_LOC ""
#endif

#ifndef DEPTH_FRAGMENT_SHADER_LOC
#define DEPTH_FRAGMENT_SHADER_LOC ""
#endif

#ifndef DEPTH_VERTEX_SHADER_LOC
#define DEPTH_VERTEX_SHADER_LOC ""
#endif

#ifndef SHADOW_MAP_FRAGMENT_SHADER_LOC
#define SHADOW_MAP_FRAGMENT_SHADER_LOC ""
#endif

#ifndef SHADOW_MAP_VERTEX_SHADER_LOC
#define SHADOW_MAP_VERTEX_SHADER_LOC ""
#endif

// Assumes that an OpenGL context has been created so that OpenGL calls can
// be made.
class OpenGLRenderer
{
   public:
      OpenGLRenderer(void);

      // Clears out the mesh id to buffer map. Deletes all existing meshes.
      void clear(void);

      // Assign the view and projection matrices based on the position and
      // orientation of the visualization camera.
      void setViewProjection(glm::mat4 & view, glm::mat4 & projection);

      void setCameraPos(const glm::vec3 & camera_pos)
      {
         camera_pos_ = camera_pos;
      }

      void setCameraLookDir(const glm::vec3 & camera_look_dir)
      {
         camera_look_dir_ = camera_look_dir;
         glm::normalize(camera_look_dir_);
      }

      void setAmbientLightStrength(const float ambient_light_strength)
      {
         ambient_light_strength_ = ambient_light_strength;
      }

      // Sets the vector from the camera position to the diffuse and shadow
      // light's position in OpenGL coordinates (y-is-up).
      void setLightDirection(const glm::vec3 & light_direction)
      {
         light_direction_ = light_direction;
      }

      void setCameraFarPlane(float camera_far_plane)
      {
         camera_far_plane_ = camera_far_plane;
      }

      void setWindowFramebufferDims(unsigned int width, unsigned int height)
      {
         window_framebuffer_width_ = width;
         window_framebuffer_height_ = height;
      }

      int addMesh(
         const data_triangleMesh_t & data_mesh,
         const float * color,
         unsigned int polygon_mode
      );

      int addMesh(
         const data_triangleMesh_t & data_mesh, unsigned int polygon_mode
      );

      void updateMesh(
         unsigned int mesh_id, const data_triangleMesh_t & data_mesh, const float * color
      );

      // Uses the default color and new vertices to change the vertices of an
      // existing mesh.
      void updateMesh(
         unsigned int mesh_id, const data_triangleMesh_t & data_mesh
      );

      // Add a series of line segments for rendering.
      int addSegments(
         const std::vector<viz::types::basic_vertex_t> & verts,
         unsigned int draw_mode
      );

      // Update the vertices of a segment that's already been submitted as a
      // renderable.
      void updateSegments(
         unsigned int segment_id,
         const std::vector<viz::types::basic_vertex_t> & verts
      );

      // Remove a renderable item. Doesn't crash if an invalid renderable ID is
      // given.
      void deleteRenderable(unsigned int renderableId);

      // Sets the model-space transformation of a set of vertices for either a
      // mesh or line segments.
      int updateModelTransform(
         unsigned int mesh_id, const glm::mat4 & transform
      );

      int updateModelColor(unsigned int mesh_id, glm::vec4 & color);

      // Draw all of the renderable items in this object.
      void draw(void);

      // Returns true if the given mesh_id is in use by the renderer, false
      // otherwise.
      bool meshValid(unsigned int mesh_id) const;

      // Returns the color of the mesh at mesh_id if the mesh is in use, else
      // it returns an all-zero color vector.
      viz::types::basic_color_t meshColor(unsigned int mesh_id) const;

      // Returns the number of vertices in the mesh at mesh_id if the mesh is
      // in use, otherwise it returns -1.
      int meshSize(unsigned int mesh_id) const;

      // Attempts to enable rendering the buffer referenced by the mesh_id.
      void enableBuffer(unsigned int mesh_id);

      // Attempts to disable rendering the buffer referenced by the mesh_id.
      void disableBuffer(unsigned int mesh_id);

      void enableShadow(unsigned int mesh_id);

      void disableShadow(unsigned int mesh_id);

      // Enables rendering of the grid.
      void enableGrid(void);

      // Disables the grid from being rendered.
      void disableGrid(void);

      unsigned int numBuffers(void) const
      {
         return mesh_id_to_buffer_.size();
      }

      unsigned int numDrawCalls(void) const
      {
         return num_draw_calls_;
      }

   private:
      // Array of vertices for updating GL buffer.
      std::vector<viz::types::basic_vertex_t> fuck_off_vert_buffer_;

      // Vertex buffer object ID.
      unsigned int vert_vbo_;

      // Vertex attribute object ID.
      unsigned int vert_vao_;

      unsigned int depth_map_fbo_;

      unsigned int depth_map_texture_;

      // Mesh shader source code.
      BasicShader mesh_shader_;

      // Line segment shader source code.
      BasicShader segment_shader_;

      // Depth shader source code.
      BasicShader depth_shader_;

      // Mapping of unique mesh IDs to mesh buffer objects.
      std::map<unsigned int, viz::FlatMeshBuffer> mesh_id_to_buffer_;

      // Tracks the ranges of the vertex buffer used by a particular mesh ID.
      std::map<unsigned int, std::pair<int, int> > mesh_id_to_vert_ranges_;

      // The IDs of the segments that are enabled for rendering.
      std::vector<unsigned int> enabled_segment_ids_;

      // The IDs of the meshes that are enabled for rendering.
      std::vector<unsigned int> enabled_mesh_ids_;

      const unsigned int shadow_map_width_;

      const unsigned int shadow_map_height_;

      unsigned int window_framebuffer_width_;

      unsigned int window_framebuffer_height_;

      // For unique-ification of mesh IDs.
      int last_mesh_id_;

      // True if the big buffers need to be updated. False otherwise.
      bool update_buffers_;

      float camera_far_plane_;

      float ambient_light_strength_;

      glm::vec3 ambient_light_color_;

      // The direction of the directional light for diffuse lighting and
      // shadow map calculation.
      glm::vec3 light_direction_;

      // The position of the camera in OpenGL coordinates. This is used for
      // rendering the shadow map.
      glm::vec3 camera_pos_;

      // The normalized camera look direction in OpenGL coordinates. This is
      // used for rendering the shadow map.
      glm::vec3 camera_look_dir_;

      // View matrix (for shader). I hate that this lives here.
      glm::mat4 view_;

      // Projection matrix (for shader). I hate that this lives here.
      glm::mat4 projection_;

      const float default_color_[4];

      // Allows the grid lines to be rendered when set to true.
      bool show_grid_;

      // Grid of lines that appears in the render window.
      Grid gridlines_;

      // The number of OpenGL draw calls made per call to this class's 'draw'
      // method.
      unsigned int num_draw_calls_;

      // Generates the EBO, VBO, and VAO OpenGL objects. Called once at
      // construction.
      void generateGlObjects(void);

      void generateDepthMapFrameBuffer(void);

      // Puts the IDs of enabled meshes into the enabled_mesh_ids_ vector and
      // puts the IDs of enabled segments into the enabled_segment_ids_ vector.
      void calculateEnabledIds(void);

      // Checks if any of the mesh buffers have their 'updated' flags set to
      // true. Sets the 'update_buffers_' bool if any of the meshes have their
      // 'updated' flags set to true. Sets all 'updated' flags for each of the
      // mesh buffers to false.
      void checkBufferUpdate(void);

      // Copies all of the data from the individual mesh buffers into the big
      // CPU-side element and vertex buffer.
      void updateBuffer(void);

      // Renders only the meshes in the scene. Assumes that the 'shader' only
      // needs the model matrices. Also assumes that the enabled_mesh_IDs_
      // vector is up to date.
      void renderMeshes(BasicShader & shader, bool shadow_map);

      // Renders only the line segments in the scene. Assumes that the 'shader'
      // only needs the model matrices. Also assumes that the
      // enabled_segment_IDs_ vector is up to date.
      void renderSegments(BasicShader & shader);

};

#endif
