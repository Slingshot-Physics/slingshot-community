#include "data_model_io.h"

#include "gauss_map.hpp"
#include "geometry_types.hpp"
#include "geometry_type_converters.hpp"
#include "mesh.hpp"
#include "mesh_ops.hpp"

#include <map>

int main(void)
{
   std::map<std::string, geometry::types::triangleMesh_t> meshes;

   meshes["cube"] = (geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::CUBE));
   meshes["capsule"] = (geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::CAPSULE));
   meshes["cylinder"] = (geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::CYLINDER));
   meshes["sphere"] = (geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::SPHERE));

   std::map<std::string, geometry::types::gaussMapMesh_t> gauss_maps;

   for (auto & mesh_iter : meshes)
   {
      gauss_maps[mesh_iter.first] = geometry::mesh::gaussMap(mesh_iter.second);
   }

   for (auto & gauss_map_iter: gauss_maps)
   {
      std::cout << "\nmesh name: " << gauss_map_iter.first << "\n";
      for (int i = 0; i < gauss_map_iter.second.numFaces; ++i)
      {
         std::cout << "\tface " << i << "\n";
         for (int j = 0; j < gauss_map_iter.second.faces[i].numTriangles; ++j)
         {
            Vector3 center = geometry::averageVertex(
               gauss_map_iter.second.numVerts, gauss_map_iter.second.verts
            );
            unsigned int tri_id = gauss_map_iter.second.faces[i].triangleStartId + j;
            Vector3 normal = geometry::calcNormal(
               gauss_map_iter.second.verts[gauss_map_iter.second.triangles[tri_id].vertIds[0]],
               gauss_map_iter.second.verts[gauss_map_iter.second.triangles[tri_id].vertIds[1]],
               gauss_map_iter.second.verts[gauss_map_iter.second.triangles[tri_id].vertIds[2]],
               center
            );
            std::cout << "\t\ttriangle " << tri_id << " normal: " << normal.unitVector() << "\n";
         }
      }

      data_gaussMapMesh_t gauss_mesh_data;
      geometry::converters::to_pod(gauss_map_iter.second, &gauss_mesh_data);

      std::string filename(gauss_map_iter.first);
      filename += ".json";
      write_data_to_file(&gauss_mesh_data, "data_gaussMapMesh_t", filename.c_str());
   }

   return 0;
}
