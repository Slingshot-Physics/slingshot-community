#ifndef GEOMETRY_VOLUME_HEADER

#include "geometry_types.hpp"

namespace geometry
{
   float volume(geometry::types::shapeCube_t cube);

   float volume(geometry::types::shapeSphere_t sphere);

   float volume(geometry::types::shapeCapsule_t capsule);

   float volume(geometry::types::shapeCylinder_t cylinder);

   float volume(geometry::types::shape_t shape);
}

#endif
