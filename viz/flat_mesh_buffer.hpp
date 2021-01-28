#ifndef FLAT_MESH_BUFFER_HEADER
#define FLAT_MESH_BUFFER_HEADER

#include <glm/glm.hpp>

#include "data_triangle_mesh.h"
#include "gl_common.hpp"
#include "viz_types.hpp"

#include <vector>

namespace viz
{

   class FlatMeshBuffer
   {
      public:
         FlatMeshBuffer(void);

         // Deletes the underlying vertex buffer and index buffer.
         void clear(void);

         // Replace the existing vertices and indices with new ones. Only
         // reallocates a buffer if the number of elements changes. Returns -1
         // if the number of vertices in 'verts' is zero.
         int update(const std::vector<viz::types::basic_vertex_t> & verts);

         // Converts the triangle mesh into a flat array of vertices. The
         // number of vertices is the number of triangles * 3. Returns -1 if
         // the number of vertices or the number of triangles in the data mesh
         // is zero.
         int update(
            const data_triangleMesh_t & data_mesh, const float * color
         );

         void updateDrawMode(GLenum mesh_draw_mode, GLenum mesh_polygon_mode)
         {
            this->draw_mode = mesh_draw_mode;
            this->polygon_mode = mesh_polygon_mode;
         }

         unsigned int numVerts(void) const
         {
            return vert_buffer.size();
         }

         // Array of vertices.
         std::vector<viz::types::basic_vertex_t> vert_buffer;

         // SE4 transformation for this mesh.
         glm::mat4 model;

         // Draw mode to use when glDrawElements is called.
         // GL_TRIANGLES, GL_LINE_LOOP, etc.
         GLenum draw_mode;

         // Draw the mesh with filled triangles or with lines.
         // GL_FILL or GL_LINES.
         GLenum polygon_mode;

         viz::types::vec4_t color;

         // Indicates that this buffer has been modified.
         bool updated;

         bool enabled;

         bool shadow_enabled;

      private:

         FlatMeshBuffer(const FlatMeshBuffer &);
         FlatMeshBuffer(FlatMeshBuffer &);
         FlatMeshBuffer & operator=(const FlatMeshBuffer &);

   };

}

#endif
