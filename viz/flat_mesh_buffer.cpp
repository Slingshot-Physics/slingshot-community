#include "flat_mesh_buffer.hpp"

#include "vector3.hpp"
#include "viztypeconverters.hpp"

#include <algorithm>

namespace viz
{

   FlatMeshBuffer::FlatMeshBuffer(void)
      : model(1.f)
      , draw_mode(GL_TRIANGLES)
      , polygon_mode(GL_FILL)
      , updated(false)
      , enabled(true)
      , shadow_enabled(true)
   {}

   int FlatMeshBuffer::update(
      const std::vector<viz::types::basic_vertex_t> & mesh_verts
   )
   {
      if (mesh_verts.size() == 0)
      {
         return -1;
      }

      vert_buffer.resize(mesh_verts.size());

      std::copy(
         mesh_verts.begin(), mesh_verts.end(), &(vert_buffer[0])
      );

      updated = true;
      return 1;
   }

   int FlatMeshBuffer::update(
      const data_triangleMesh_t & data_mesh, const float * color
   )
   {
      if (data_mesh.numTriangles == 0 || data_mesh.numVerts == 0)
      {
         return -1;
      }

      vert_buffer.resize(data_mesh.numTriangles * 3);

      // For each triangle
      for (unsigned int i = 0; i < data_mesh.numTriangles; ++i)
      {
         // For each vertex on the triangle
         for (unsigned int j = 0; j < 3; ++j)
         {
            viz::types::basic_vertex_t temp_vert_attrib;
            for (unsigned int k = 0; k < 3; ++k)
            {
               temp_vert_attrib.normal[k] = data_mesh.triangles[i].normal.v[k];
               temp_vert_attrib.pos[k] = data_mesh.verts[data_mesh.triangles[i].vertIds[j]].v[k];
            }

            if (color != nullptr)
            {
               for (unsigned int k = 0; k < 4; ++k)
               {
                  temp_vert_attrib.color[k] = color[k];
               }
            }

            vert_buffer[3 * i + j] = temp_vert_attrib;
         }
      }

      updated = true;

      return 1;
   }

}
