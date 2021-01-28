#ifndef GAUSS_MAP_HEADER
#define GAUSS_MAP_HEADER

#include "data_enums.h"
#include "data_gauss_map_mesh.h"
#include "geometry_types.hpp"

namespace geometry
{

namespace mesh
{
   data_gaussMapMesh_t loadGaussMapData(
      geometry::types::enumShape_t shape_type
   );

   data_gaussMapMesh_t loadGaussMapData(data_shapeType_t shape_type);

   geometry::types::gaussMapMesh_t loadGaussMap(
      geometry::types::enumShape_t shape_type
   );

   // Generates the collider-space polygon of vertices on the perimeter of a
   // given face ID on the gauss map mesh.
   geometry::types::polygon50_t polygonFromFace(
      const geometry::types::gaussMapMesh_t & gauss_mesh_M,
      unsigned int face_id
   );

   geometry::types::gaussMapMesh_t gaussMap(
      const geometry::types::triangleMesh_t & convex_mesh
   );
}

}

#endif
