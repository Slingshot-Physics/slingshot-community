#include "opengl_renderer.hpp"

#include <algorithm>
#include <iostream>

OpenGLRenderer::OpenGLRenderer(void)
   : mesh_shader_(
      SHADOW_MAP_VERTEX_SHADER_LOC, SHADOW_MAP_FRAGMENT_SHADER_LOC
   )
   , segment_shader_(
      VERTEX_MVP_SHADER_LOC, LINE_FRAGMENT_SHADER_LOC
   )
   , depth_shader_(
      DEPTH_VERTEX_SHADER_LOC, DEPTH_FRAGMENT_SHADER_LOC
   )
   , shadow_map_width_(2048)
   , shadow_map_height_(2048)
   , window_framebuffer_width_(800)
   , window_framebuffer_height_(600)
   , last_mesh_id_(0)
   , update_buffers_(true)
   , camera_far_plane_(100.f)
   , ambient_light_strength_(0.2f)
   , ambient_light_color_(1.f)
   , light_direction_(6.f, 12.f, 5.f)
   , default_color_{1.f, 0.6902f, 0.f, 1.f}
   , show_grid_(true)
   , gridlines_(
      INSTANCED_VERTEX_SHADER_LOC,
      EMITTER_FRAGMENT_SHADER_LOC,
      1.0f
   )
   , num_draw_calls_(0)
{
   fuck_off_vert_buffer_.reserve(FUCKOFFBUFFERSIZE);

   generateGlObjects();
   generateDepthMapFrameBuffer();
}

void OpenGLRenderer::clear(void)
{
   mesh_id_to_buffer_.clear();
   mesh_id_to_vert_ranges_.clear();
   last_mesh_id_ = 0;
   num_draw_calls_ = 0;
}

void OpenGLRenderer::setViewProjection(glm::mat4 & view, glm::mat4 & projection)
{
   view_ = view;
   projection_ = projection;

   gridlines_.setViewProjection(view, projection);
}

int OpenGLRenderer::addMesh(
   const data_triangleMesh_t & data_mesh,
   const float * color,
   unsigned int polygon_mode
)
{
   int new_mesh_id = last_mesh_id_;
   viz::FlatMeshBuffer & buf_ref = mesh_id_to_buffer_[new_mesh_id];

   buf_ref.update(data_mesh, color == nullptr ? default_color_ : color);
   buf_ref.draw_mode = GL_TRIANGLES;
   buf_ref.polygon_mode = polygon_mode;

   ++last_mesh_id_;

   return new_mesh_id;
}

int OpenGLRenderer::addMesh(const data_triangleMesh_t & data_mesh, unsigned int polygon_mode)
{
   return addMesh(data_mesh, default_color_, polygon_mode);
}

void OpenGLRenderer::updateMesh(
   unsigned int mesh_id, const data_triangleMesh_t & data_mesh, const float * color
)
{
   if (!meshValid(mesh_id))
   {
      std::cout << "Couldn't find mesh ID: " << mesh_id << " to update\n";
      return;
   }

   viz::FlatMeshBuffer & buf_ref = mesh_id_to_buffer_[mesh_id];
   buf_ref.update(data_mesh, (color == nullptr) ? default_color_ : color);
}

void OpenGLRenderer::updateMesh(
   unsigned int mesh_id, const data_triangleMesh_t & data_mesh
)
{
   updateMesh(mesh_id, data_mesh, nullptr);
}

int OpenGLRenderer::addSegments(
   const std::vector<viz::types::basic_vertex_t> & verts,
   unsigned int draw_mode
)
{
   std::cout << "Adding " << verts.size() << " segment points\n";
   int new_mesh_id = last_mesh_id_;
   viz::FlatMeshBuffer & buf_ref = mesh_id_to_buffer_[new_mesh_id];

   buf_ref.update(verts);
   buf_ref.updateDrawMode(draw_mode, GL_LINE);

   ++last_mesh_id_;

   return new_mesh_id;
}

void OpenGLRenderer::updateSegments(
   unsigned int segment_id,
   const std::vector<viz::types::basic_vertex_t> & verts
)
{
   if (!meshValid(segment_id))
   {
      std::cout << "Couldn't find segment - invalid segment ID " << segment_id << "\n";
      return;
   }

   viz::FlatMeshBuffer & buf_ref = mesh_id_to_buffer_[segment_id];
   buf_ref.update(verts);

   buf_ref.updated = true;
}

void OpenGLRenderer::deleteRenderable(unsigned int renderable_id)
{
   mesh_id_to_buffer_.erase(renderable_id);
}

int OpenGLRenderer::updateModelTransform(
   unsigned int mesh_id, const glm::mat4 & transform
)
{
   if (!meshValid(mesh_id))
   {
      std::cout << "Couldn't update transform - invalid mesh ID " << mesh_id << "\n";
      return -1;
   }

   mesh_id_to_buffer_[mesh_id].model = transform;
   return 1;
}

int OpenGLRenderer::updateModelColor(unsigned int mesh_id, glm::vec4 & color)
{
   if (!meshValid(mesh_id))
   {
      std::cout << "Couldn't update color - invalid mesh ID " << mesh_id << "\n";
      return -1;
   }

   viz::FlatMeshBuffer & buf_ref = mesh_id_to_buffer_[mesh_id];
   for (unsigned int i = 0; i < buf_ref.numVerts(); ++i)
   {
      for (unsigned int j = 0; j < 4; ++j)
      {
         buf_ref.vert_buffer[i].color[j] = color[j];
      }
   }

   buf_ref.updated = true;

   return 1;
}

void OpenGLRenderer::draw(void)
{
   if (mesh_id_to_buffer_.size() == 0)
   {
      if (show_grid_)
      {
         gridlines_.draw();
      }
      return;
   }

   checkBufferUpdate();

   if (update_buffers_)
   {
      calculateEnabledIds();
      updateBuffer();

      glBindBuffer(GL_ARRAY_BUFFER, vert_vbo_);
      glBufferSubData(
         GL_ARRAY_BUFFER,
         0,
         FUCKOFFBUFFERSIZE * sizeof(viz::types::basic_vertex_t),
         &(fuck_off_vert_buffer_[0])
      );
      glBindBuffer(GL_ARRAY_BUFFER, 0);

      update_buffers_ = false;
   }

   num_draw_calls_ = 0;

   float camera_radius = camera_far_plane_ / 2.f;

   // Use the depth map shader for this portion of the scene rendering.
   glm::mat4 light_projection = glm::ortho(
      -camera_radius,
      camera_radius,
      -camera_radius,
      camera_radius,
      0.1f,
      camera_radius * 2.f
   );

   glm::vec3 light_pos = camera_pos_ + camera_look_dir_ * camera_radius + glm::normalize(light_direction_) * camera_radius;

   glm::mat4 light_view = glm::lookAt(
      light_pos, camera_pos_ + camera_look_dir_ * camera_radius, glm::vec3(0.f, 0.f, 1.f)
   );

   glm::mat4 light_space_matrix = light_projection * light_view;

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   depth_shader_.use();
   depth_shader_.setUniformMatrix4("light_space_matrix", light_space_matrix);

   glViewport(0, 0, shadow_map_width_, shadow_map_height_);
   glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo_);
   glClear(GL_DEPTH_BUFFER_BIT);
   glCullFace(GL_FRONT);

   renderMeshes(depth_shader_, true);

   glCullFace(GL_BACK);

   glBindFramebuffer(GL_FRAMEBUFFER, 0);

   glViewport(0, 0, window_framebuffer_width_, window_framebuffer_height_);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   mesh_shader_.use();
   mesh_shader_.setUniformMatrix4("view", view_);
   mesh_shader_.setUniformMatrix4("projection", projection_);
   mesh_shader_.setUniformMatrix4("lightSpaceMatrix", light_space_matrix);
   mesh_shader_.setUniformVec3("ambientLightColor", ambient_light_color_);
   mesh_shader_.setUniformVec3("diffuseLightDir", light_direction_);
   mesh_shader_.setUniformVec3("shadowLightDir", light_direction_);
   mesh_shader_.setUniformFloat("ambientLightStrength", ambient_light_strength_);

   glBindTexture(GL_TEXTURE_2D, depth_map_texture_);

   renderMeshes(mesh_shader_, false);

   segment_shader_.use();
   segment_shader_.setUniformMatrix4("view", view_);
   segment_shader_.setUniformMatrix4("projection", projection_);

   renderSegments(segment_shader_);

   if (show_grid_)
   {
      gridlines_.draw();
   }

}

bool OpenGLRenderer::meshValid(unsigned int mesh_id) const
{
   return mesh_id_to_buffer_.find(mesh_id) != mesh_id_to_buffer_.end();
}

viz::types::basic_color_t OpenGLRenderer::meshColor(unsigned int mesh_id) const
{
   viz::types::basic_color_t color;
   for (int i = 0; i < 4; ++i)
   {
      color[i] = 0.f;
   }

   if (!meshValid(mesh_id))
   {
      std::cout << "Couldn't find mesh ID " << mesh_id << "\n";
      return color;
   }

   const viz::FlatMeshBuffer & temp_buffer = mesh_id_to_buffer_.at(mesh_id);

   if (temp_buffer.numVerts() == 0)
   {
      return color;
   }

   color[0] = temp_buffer.vert_buffer[0].color[0];
   color[1] = temp_buffer.vert_buffer[0].color[1];
   color[2] = temp_buffer.vert_buffer[0].color[2];
   color[3] = temp_buffer.vert_buffer[0].color[3];

   return color;
}

int OpenGLRenderer::meshSize(unsigned int mesh_id) const
{
   if (!meshValid(mesh_id))
   {
      return -1;
   }

   return static_cast<int>(mesh_id_to_buffer_.at(mesh_id).numVerts());
}

void OpenGLRenderer::enableBuffer(unsigned int mesh_id)
{
   if (!meshValid(mesh_id))
   {
      std::cout << "Couldn't find mesh ID " << mesh_id << "\n";
      return;
   }

   mesh_id_to_buffer_.at(mesh_id).enabled = true;
   mesh_id_to_buffer_.at(mesh_id).updated = true;
}

void OpenGLRenderer::disableBuffer(unsigned int mesh_id)
{
   if (!meshValid(mesh_id))
   {
      std::cout << "Couldn't find mesh ID " << mesh_id << "\n";
      return;
   }

   mesh_id_to_buffer_.at(mesh_id).enabled = false;
   mesh_id_to_buffer_.at(mesh_id).updated = true;
}

void OpenGLRenderer::enableShadow(unsigned int mesh_id)
{
   if (!meshValid(mesh_id))
   {
      std::cout << "Couldn't find mesh ID " << mesh_id << "\n";
      return;
   }

   mesh_id_to_buffer_.at(mesh_id).shadow_enabled = true;
   mesh_id_to_buffer_.at(mesh_id).updated = true;
}

void OpenGLRenderer::disableShadow(unsigned int mesh_id)
{
   if (!meshValid(mesh_id))
   {
      std::cout << "Couldn't find mesh ID " << mesh_id << "\n";
      return;
   }

   mesh_id_to_buffer_.at(mesh_id).shadow_enabled = false;
   mesh_id_to_buffer_.at(mesh_id).updated = true;
}

void OpenGLRenderer::enableGrid(void)
{
   show_grid_ = true;
}

void OpenGLRenderer::disableGrid(void)
{
   show_grid_ = false;
}

void OpenGLRenderer::generateGlObjects(void)
{
   glGenVertexArrays(1, &vert_vao_);
   glGenBuffers(1, &vert_vbo_);

   glBindVertexArray(vert_vao_);
   glBindBuffer(GL_ARRAY_BUFFER, vert_vbo_);
   glBufferData(
      GL_ARRAY_BUFFER,
      sizeof(viz::types::basic_vertex_t) * FUCKOFFBUFFERSIZE,
      nullptr,
      GL_DYNAMIC_DRAW
   );

   // Specific to my shaders and vertex layouts, so there's kind of some
   // coupling between the shader source code and the vertex layout.
   glVertexAttribPointer(
      0, 3, GL_FLOAT, GL_FALSE, sizeof(viz::types::basic_vertex_t), (void*)0
   );
   glEnableVertexAttribArray(0);

   glVertexAttribPointer(
      1, 4, GL_FLOAT, GL_FALSE, sizeof(viz::types::basic_vertex_t), (void*)(3*sizeof(float))
   );
   glEnableVertexAttribArray(1);

   glVertexAttribPointer(
      2, 3, GL_FLOAT, GL_FALSE, sizeof(viz::types::basic_vertex_t), (void*)(10*sizeof(float))
   );
   glEnableVertexAttribArray(2);

   glBindVertexArray(0);
}

void OpenGLRenderer::generateDepthMapFrameBuffer(void)
{
   glGenFramebuffers(1, &depth_map_fbo_);

   glGenTextures(1, &depth_map_texture_);
   glBindTexture(GL_TEXTURE_2D, depth_map_texture_);
   glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_DEPTH_COMPONENT,
      shadow_map_width_,
      shadow_map_height_,
      0,
      GL_DEPTH_COMPONENT,
      GL_FLOAT,
      nullptr
   );
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
   float borderColor[] = {1.f, 1.f, 1.f};
   glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
   glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo_);
   glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map_texture_, 0
   );
   glDrawBuffer(GL_NONE);
   glReadBuffer(GL_NONE);

   GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

   if (status != GL_FRAMEBUFFER_COMPLETE)
   {
      std::cout << "couldn't work with the framebuffer somehow? " << status << "\n";
   }

   glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OpenGLRenderer::calculateEnabledIds(void)
{
   enabled_segment_ids_.clear();
   enabled_segment_ids_.reserve(mesh_id_to_buffer_.size());

   enabled_mesh_ids_.clear();
   enabled_mesh_ids_.reserve(mesh_id_to_buffer_.size());

   for (const auto & mesh_it : mesh_id_to_buffer_)
   {
      if (!mesh_it.second.enabled)
      {
         continue;
      }
      
      if (mesh_it.second.draw_mode == GL_TRIANGLES)
      {
         enabled_mesh_ids_.push_back(mesh_it.first);
      }
      else
      {
         enabled_segment_ids_.push_back(mesh_it.first);
      }
   }
}

// If the mesh buffers have been updated, then the big CPU-side buffers will
// need to be updated, and the big GPU buffers will also need to be updated.
void OpenGLRenderer::checkBufferUpdate(void)
{
   for (auto & mesh_it : mesh_id_to_buffer_)
   {
      update_buffers_ |= mesh_it.second.updated;
      mesh_it.second.updated = false;
   }
}

void OpenGLRenderer::updateBuffer(void)
{
   mesh_id_to_vert_ranges_.clear();
   unsigned int vert_counter = 0;

   for (const auto & mesh_it : mesh_id_to_buffer_)
   {
      const unsigned int & mesh_id = mesh_it.first;
      const unsigned int & num_verts = mesh_it.second.numVerts();

      if (vert_counter + num_verts >= FUCKOFFBUFFERSIZE)
      {
         std::cout << "Too many things are being drawn! Mesh ID " << mesh_it.first << " broke the camel's back\n";
         break;
      }

      std::copy(
         mesh_it.second.vert_buffer.begin(),
         mesh_it.second.vert_buffer.end(),
         &(fuck_off_vert_buffer_[vert_counter])
      );

      mesh_id_to_vert_ranges_[mesh_id].first = vert_counter;
      mesh_id_to_vert_ranges_[mesh_id].second = (
         vert_counter + num_verts - 1
      );

      vert_counter += num_verts;
   }
}

void OpenGLRenderer::renderMeshes(BasicShader & shader, bool shadow_map)
{
   glBindVertexArray(vert_vao_);

   for (const auto mesh_id : enabled_mesh_ids_)
   {
      const auto & mesh = mesh_id_to_buffer_[mesh_id];
      if (shadow_map && !mesh.shadow_enabled)
      {
         continue;
      }

      shader.setUniformMatrix4("model", mesh.model);
      glPolygonMode(GL_FRONT_AND_BACK, mesh.polygon_mode);

      std::pair<unsigned int, unsigned int> range = \
         mesh_id_to_vert_ranges_[mesh_id];

      glDrawArrays(mesh.draw_mode, range.first, mesh.numVerts());
      ++num_draw_calls_;
   }

   glBindVertexArray(0);
}

void OpenGLRenderer::renderSegments(BasicShader & shader)
{
   glBindVertexArray(vert_vao_);

   for (const auto segment_id : enabled_segment_ids_)
   {
      auto & segment = mesh_id_to_buffer_[segment_id];
      shader.setUniformMatrix4("model", segment.model);
      glPolygonMode(GL_FRONT_AND_BACK, segment.polygon_mode);

      std::pair<unsigned int, unsigned int> range = \
         mesh_id_to_vert_ranges_[segment_id];

      ++num_draw_calls_;
      glDrawArrays(segment.draw_mode, range.first, segment.numVerts());
   }

   glBindVertexArray(0);
}
