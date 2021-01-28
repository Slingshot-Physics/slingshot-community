#include "epa_types.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include "mesh_ops.hpp"
#include "geometry_type_converters.hpp"

namespace geometry
{

namespace types
{

namespace epa
{
   epaMesh_t::epaMesh_t(void)
   {

   }

   epaMesh_t::epaMesh_t(
      const geometry::types::minkowskiDiffSimplex_t & tetra_simplex
   )
   {
      initialize(tetra_simplex);
   }

   void epaMesh_t::initialize(
      const geometry::types::minkowskiDiffSimplex_t & tetra_simplex
   )
   {
      triangles.clear();
      for (unsigned int i = 0; i < 4; ++i)
      {
         mdVerts[i].bodyAVertId = tetra_simplex.bodyAVertIds[i];
         mdVerts[i].bodyBVertId = tetra_simplex.bodyBVertIds[i];
         mdVerts[i].vert = tetra_simplex.verts[i];
         mdVerts[i].bodyAVert = tetra_simplex.bodyAVerts[i];
         mdVerts[i].bodyBVert = tetra_simplex.bodyBVerts[i];
      }

      triangles.add_triangle(0, 1, 3);
      triangles.add_triangle(0, 2, 3);
      triangles.add_triangle(0, 1, 2);
      triangles.add_triangle(1, 2, 3);

      numMdVerts = 4;
   }

   void epaMesh_t::print(void) const
   {
      std::cout << "EPA mesh data\n";
      std::cout << "EPA mesh vertices\n";
      for (unsigned int i = 0; i < numMdVerts; ++i)
      {
         std::cout << "\t" << mdVerts[i].vert << "\n";
         std::cout << "\tbody A id: " << mdVerts[i].bodyAVertId << "\n";
         std::cout << "\tbody B id: " << mdVerts[i].bodyBVertId << "\n";
      }
      std::cout << "EPA mesh triangles\n";
      for (unsigned int i = 0; i < triangles.size(); ++i)
      {
         print(i);
      }
      std::cout << std::endl;
   }

   void epaMesh_t::print(unsigned int id) const
   {
      std::cout << "triangle id: " << id << "\n";
      std::cout << "\tvert ids:\n";
      for (unsigned int i = 0; i < 3; ++i)
      {
         std::cout << "\t" << triangles[id].vertIds[i] << ": " << mdVerts[triangles[id].vertIds[i]].vert << "\n";
      }
      Vector3 normal = (
         mdVerts[triangles[id].vertIds[0]].vert - mdVerts[triangles[id].vertIds[1]].vert
         ).crossProduct(
            mdVerts[triangles[id].vertIds[2]].vert - mdVerts[triangles[id].vertIds[1]].vert
         );

      std::cout << "normal: " << normal << std::endl;

      std::cout << "\tneighbor ids:\n";
      for (unsigned int i = 0; i < 3; ++i)
      {
         std::cout << "\t" << i << " " << triangles[id].neighborIds[i] << std::endl;
      }
   }

   bool epaMesh_t::vertExists(
      const geometry::types::minkowskiDiffVertex_t & md_vert
   )
   {
      // Filter out new vertices based on body IDs, if possible, otherwise
      // filter them out based on distance.
      for (unsigned int i = 0; i < numMdVerts; ++i)
      {
         // if (md_vert_a.vertId != -1 && md_vert_b.vertId != -1)
         if (md_vert.bodyAVertId != -1 && md_vert.bodyBVertId != -1)
         {
            if (
               (mdVerts[i].bodyAVertId == md_vert.bodyAVertId) &&
               (mdVerts[i].bodyBVertId == md_vert.bodyBVertId)
            )
            {
               return true;
            }
         }
         else
         {
            if ((mdVerts[i].vert - md_vert.vert).magnitudeSquared() < 1e-14f)
            {
               return true;
            }
         }
      }

      return false;
   }

   int epaMesh_t::addVert(
      const geometry::types::minkowskiDiffVertex_t & md_vert
   )
   {
      if (numMdVerts >= 3 * MAX_SIZE)
      {
         return -2;
      }

      mdVerts[numMdVerts].vert = md_vert.vert;
      mdVerts[numMdVerts].bodyAVertId = md_vert.bodyAVertId;
      mdVerts[numMdVerts].bodyBVertId = md_vert.bodyBVertId;
      mdVerts[numMdVerts].bodyAVert = md_vert.bodyAVert;
      mdVerts[numMdVerts].bodyBVert = md_vert.bodyBVert;

      ++numMdVerts;

      return numMdVerts - 1;
   }

   void epaMesh_t::to_triangle_mesh(data_triangleMesh_t * data_out)
   {
      data_out->numVerts = numMdVerts;

      Vector3 center;
      for (unsigned int i = 0; i < numMdVerts; ++i)
      {
         geometry::converters::to_pod(mdVerts[i].vert, &(data_out->verts[i]));
         center += mdVerts[i].vert / numMdVerts;
      }

      data_out->numTriangles = triangles.size();
      for (unsigned int i = 0; i < triangles.size(); ++i)
      {
         for (int j = 0; j < 3; ++j)
         {
            data_out->triangles[i].vertIds[j] = triangles[i].vertIds[j];
         }

         Vector3 normal = calcNormal(
            mdVerts[triangles[i].vertIds[0]].vert,
            mdVerts[triangles[i].vertIds[1]].vert,
            mdVerts[triangles[i].vertIds[2]].vert,
            center
         );

         geometry::converters::to_pod(normal, &(data_out->triangles[i].normal));
      }
   }

} // epa

} // types

} // geometry
