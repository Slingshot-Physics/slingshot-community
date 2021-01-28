#ifndef GEOMETRY_INERTIA_HEADER
#define GEOMETRY_INERTIA_HEADER

#include "geometry_types.hpp"

// These assume that shapes have a constant mass density, so the total moment
// of inertia can be obtained by multiplying by the shape's mass.
namespace geometry
{
   // Calculates the moment of inertia tensor of a cube about its center of
   // mass in the cube's body frame with the cube oriented as:
   //   length along body x-axis
   //   width along body y-axis
   //   height along body z-axis
   Matrix33 inertiaTensor(geometry::types::shapeCube_t cube);

   // Calculates the moment of inertia tensor of a sphere about its center of
   // mass in the sphere's body frame.
   Matrix33 inertiaTensor(geometry::types::shapeSphere_t sphere);

   // Calculates the moment of inertia tensor of a capsule about its center of
   // mass in the capsule's body frame with the capsule oriented as:
   //   radius in the body x-y plane
   //   height along the body z-axis
   Matrix33 inertiaTensor(geometry::types::shapeCapsule_t capsule);

   // Calculates the moment of inertia tensor of a cylinder about its center of
   // mass in the cylinder's body frame with the cylinder oriented as:
   //   radius in the body x-y plane
   //   height along the body z-axis
   Matrix33 inertiaTensor(geometry::types::shapeCylinder_t cylinder);

   // Calculates the moment of inertia tensor of the active shape about the
   // shape's center of mass in the shape's body frame.
   Matrix33 inertiaTensor(geometry::types::shape_t shape);
}

#endif
