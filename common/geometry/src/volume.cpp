#include "volume.hpp"

namespace geometry
{
   float volume(geometry::types::shapeCube_t cube)
   {
      return cube.length * cube.width * cube.height;
   }

   float volume(geometry::types::shapeSphere_t sphere)
   {
      return 4.f * M_PI * std::pow(sphere.radius, 3.f) / 3.f;
   }

   float volume(geometry::types::shapeCapsule_t capsule)
   {
      return M_PI * (4.f * std::pow(capsule.radius, 3.f) / 3.f + std::pow(capsule.radius, 2.f) * capsule.height);
   }

   float volume(geometry::types::shapeCylinder_t cylinder)
   {
      return M_PI * std::pow(cylinder.radius, 2.f) * cylinder.height;
   }

   float volume(geometry::types::shape_t shape)
   {
      float val = 0.f;
      switch(shape.shapeType)
      {
         case geometry::types::enumShape_t::CUBE:
            val = volume(shape.cube);
            break;
         case geometry::types::enumShape_t::SPHERE:
            val = volume(shape.sphere);
            break;
         case geometry::types::enumShape_t::CAPSULE:
            val = volume(shape.capsule);
            break;
         case geometry::types::enumShape_t::CYLINDER:
            val = volume(shape.cylinder);
            break;
         case geometry::types::enumShape_t::NONE:
            break;
      }

      return val;
   }
}
