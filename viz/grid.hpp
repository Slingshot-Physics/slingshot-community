#ifndef GRID_HEADER
#define GRID_HEADER

#include <array>
#include <cstring>

#include "basic_shader.hpp"
#include "viztypeconverters.hpp"

class Grid
{
   public:

      // Creates a grid of intersecting lines around the origin.
      Grid(
         const char * vsSourceLoc,
         const char * fsSourceLoc,
         float spacing
      )
         : model_(glm::mat4(1.0f))
         , view_(glm::mat4(1.0f))
         , projection_(glm::mat4(1.0f))
         , shader_(vsSourceLoc, fsSourceLoc)
         , vertVao_(0)
         , vertVbo_(0)
         , instanceVbo_(0)
         , spacing_(spacing)
      {
         int num_offsets = static_cast<int>(offsets_.size());
         for (int i = 0; i < num_offsets; ++i)
         {
            float delta = i - (num_offsets - 1) / 2;
            offsets_[i] = spacing_ * static_cast<float>(delta);
         }

         std::memset(vertices_.data(), 0, sizeof(vertices_[0]) * 4);

         vertices_[0].pos[0] = 32.0 * spacing_;
         vertices_[0].color[0] = 0.7;
         vertices_[0].color[1] = 0.7;
         vertices_[0].color[2] = 0.7;
         vertices_[0].color[3] = 0.7;
         vertices_[0].normal[2] = 1.0;

         vertices_[1].pos[0] = -32.0 * spacing_;
         vertices_[1].color[0] = 0.7;
         vertices_[1].color[1] = 0.7;
         vertices_[1].color[2] = 0.7;
         vertices_[1].color[3] = 0.7;
         vertices_[1].normal[2] = 1.0;

         vertices_[2].pos[2] = 32.0 * spacing_;
         vertices_[2].color[0] = 0.7;
         vertices_[2].color[1] = 0.7;
         vertices_[2].color[2] = 0.7;
         vertices_[2].color[3] = 0.7;
         vertices_[2].normal[0] = 1.0;

         vertices_[3].pos[2] = -32.0 * spacing_;
         vertices_[3].color[0] = 0.7;
         vertices_[3].color[1] = 0.7;
         vertices_[3].color[2] = 0.7;
         vertices_[3].color[3] = 0.7;
         vertices_[3].normal[0] = 1.0;

         generateVertBufferVao_();
      }

      // Makes two draw calls to the instances of the perpendicular lines.
      void draw(void)
      {
         shader_.use();

         glBindVertexArray(vertVao_);

         shader_.setUniformMatrix4("model", model_);
         shader_.setUniformMatrix4("view", view_);
         shader_.setUniformMatrix4("projection", projection_);
         glDrawArraysInstanced(GL_LINE_STRIP, 0, 2, offsets_.size());
         glDrawArraysInstanced(GL_LINE_STRIP, 2, 2, offsets_.size());

         glBindVertexArray(0);
      }

      void setViewProjection(
         const glm::mat4 & view, const glm::mat4 & projection
      )
      {
         this->view_ = view;
         this->projection_ = projection;
      }

   private:
      glm::mat4 model_;

      glm::mat4 view_;

      glm::mat4 projection_;

      BasicShader shader_;

      std::array<viz::types::basic_vertex_t, 4> vertices_;

      unsigned int vertVao_;

      unsigned int vertVbo_;

      unsigned int instanceVbo_;

      // The offsets of the grid lines being drawn.
      std::array<float, 65> offsets_;

      float spacing_;

      void generateVertBufferVao_(void)
      {
         glGenBuffers(1, &instanceVbo_);
         glBindBuffer(GL_ARRAY_BUFFER, instanceVbo_);
         glBufferData(
            GL_ARRAY_BUFFER,
            sizeof(float) * offsets_.size(),
            offsets_.data(),
            GL_STATIC_DRAW
         );
         glBindBuffer(GL_ARRAY_BUFFER, 0);

         glGenBuffers(1, &vertVbo_);
         glGenVertexArrays(1, &vertVao_);

         glBindVertexArray(vertVao_);
         glBindBuffer(GL_ARRAY_BUFFER, vertVbo_);
         glBufferData(
            GL_ARRAY_BUFFER,
            sizeof(viz::types::basic_vertex_t) * 4,
            vertices_.data(),
            GL_STATIC_DRAW
         );

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

         glEnableVertexAttribArray(3);
         glBindBuffer(GL_ARRAY_BUFFER, instanceVbo_);
         glVertexAttribPointer(
            3, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0
         );
         glBindBuffer(GL_ARRAY_BUFFER, 0);
         glVertexAttribDivisor(3, 1);

         glBindVertexArray(0);
      }
};

#endif
