#include "inertia.hpp"

#include "volume.hpp"

namespace geometry
{
   Matrix33 inertiaTensor(geometry::types::shapeCube_t cube)
   {
      Vector3 inertia_diag;

      inertia_diag[0] = (1.f / 12.f) * (
         cube.width * cube.width + cube.height * cube.height
      );
      inertia_diag[1] = (1.f / 12.f) * (
         cube.length * cube.length + cube.height * cube.height
      );
      inertia_diag[2] = (1.f / 12.f) * (
         cube.width * cube.width + cube.length * cube.length
      );

      Matrix33 J = Matrix33::diag(inertia_diag);
      return J;
   }

   Matrix33 inertiaTensor(geometry::types::shapeSphere_t sphere)
   {
      const float all_vals = (2.f / 5.f) * (sphere.radius * sphere.radius);
      Vector3 inertia_diag(all_vals, all_vals, all_vals);

      Matrix33 J = Matrix33::diag(inertia_diag);
      return J;
   }

   Matrix33 inertiaTensor(geometry::types::shapeCapsule_t capsule)
   {
      const float density = (1.f / volume(capsule));

      const geometry::types::shapeSphere_t temp_sphere = {capsule.radius};
      const float mass_hemispheres = volume(temp_sphere) * density;

      const geometry::types::shapeCylinder_t temp_cylinder = {capsule.radius, capsule.height};
      const float mass_cylinder = volume(temp_cylinder) * density;

      const float r2 = capsule.radius * capsule.radius;
      const float h2 = capsule.height * capsule.height;
      const float rh = capsule.radius * capsule.height;

      Vector3 inertia_diag;

      inertia_diag[0] = mass_cylinder * (h2 / 12.f + r2 / 4.f) + mass_hemispheres * (2.f * r2 / 5.f + h2 / 2.f + 3.f * rh / 8.f);
      inertia_diag[1] = inertia_diag[0];
      inertia_diag[2] = mass_cylinder * r2 / 2.f + mass_hemispheres * 2.f * r2 / 5.f;

      Matrix33 J = Matrix33::diag(inertia_diag);
      return J;
   }

   Matrix33 inertiaTensor(geometry::types::shapeCylinder_t cylinder)
   {
      const float r2 = cylinder.radius * cylinder.radius;
      const float h2 = cylinder.height * cylinder.height;

      Vector3 inertia_diag;

      inertia_diag[0] = (1.f / 12.f) * (3.f * r2 + h2);
      inertia_diag[1] = inertia_diag[0];
      inertia_diag[2] = 0.5f * r2;

      Matrix33 J = Matrix33::diag(inertia_diag);
      return J;
   }

   Matrix33 inertiaTensor(geometry::types::shape_t shape)
   {
      Matrix33 J;
      switch(shape.shapeType)
      {
         case geometry::types::enumShape_t::CUBE:
            J = inertiaTensor(shape.cube);
            break;
         case geometry::types::enumShape_t::SPHERE:
            J = inertiaTensor(shape.sphere);
            break;
         case geometry::types::enumShape_t::CAPSULE:
            J = inertiaTensor(shape.capsule);
            break;
         case geometry::types::enumShape_t::CYLINDER:
            J = inertiaTensor(shape.cylinder);
            break;
         case geometry::types::enumShape_t::NONE:
            break;
      }

      return J;
   }
}
