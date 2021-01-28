#ifndef AABB_HULL_HEADER
#define AABB_HULL_HEADER

#include "geometry_types.hpp"

namespace geometry
{
   // Calculate an AABB hull for a convex polyhedron based on its scale,
   // rotation, and translation from the mesh coordinates. Note that the
   // convex polyhedron is in collider space, and the transform is from
   // collider space to world coordinates.
   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::convexPolyhedron_t & conv_polyhedron_C
   );

   // Calculate an AABB hull for a cube based on the cube's position.
   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCube_t & cube
   );

   // Calculate an AABB hull for a sphere based on the sphere's position.
   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeSphere_t & sphere
   );

   // Calculate an AABB hull for a capsule based on the capsule's position.
   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCapsule_t & capsule
   );

   // Calculate an AABB hull for a cylinder based on the cylinder's position.
   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCylinder_t & cylinder
   );

   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shape_t & shape
   );

   // Calculate an AABB hull for a convex polyhedron based on its scale,
   // rotation, and translation from the mesh coordinates. Note that the
   // convex polyhedron is in collider space, and the transform is from
   // collider space to world coordinates.
   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::convexPolyhedron_t & conv_polyhedron_C
   );

   // Calculate an AABB hull for a cube based on the cube's position.
   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCube_t & cube
   );

   // Calculate an AABB hull for a sphere based on the sphere's position.
   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeSphere_t & sphere
   );

   // Calculate an AABB hull for a capsule based on the capsule's position.
   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCapsule_t & capsule
   );

   // Calculate an AABB hull for a cylinder based on the cylinder's position.
   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCylinder_t & cylinder
   );

   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shape_t & shape
   );
}

#endif
