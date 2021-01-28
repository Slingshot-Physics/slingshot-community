#ifndef RAYCAST_HEADER
#define RAYCAST_HEADER

#include "geometry_types.hpp"

namespace geometry
{
   // The ray is formulated as:
   //    r(u) = ray_start + u * (ray_end - ray_start)
   // So that r(u=1) = ray_end. ray_end can be any point on the ray.
   // Returns a raycast result with the number of points of intersection
   // between the ray and the sphere, an array of hit points (the lowest index
   // is the hit point nearest to ray_start), and an array of the ray
   // parameters leading to the hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeSphere_t & sphere_C
   );

   // The ray is formulated as:
   //    r(u) = ray_start + u * (ray_end - ray_start)
   // So that r(u=1) = ray_end. ray_end can be any point on the ray.
   // Returns a raycast result with the number of points of intersection
   // between the ray and the cylinder, an array of hit points (the lowest
   // index is the hit point nearest to ray_start), and an array of the ray
   // parameters leading to the hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCylinder_t & cylinder_C
   );

   // The ray is formulated as:
   //    r(u) = ray_start + u * (ray_end - ray_start)
   // So that r(u=1) = ray_end. ray_end can be any point on the ray.
   // Returns a raycast result with the number of points of intersection
   // between the ray and the capsule, an array of hit points (the lowest
   // index is the hit point nearest to ray_start), and an array of the ray
   // parameters leading to the hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCapsule_t & capsule_C
   );

   // The ray is formulated as:
   //    r(u) = ray_start + u * (ray_end - ray_start)
   // So that r(u=1) = ray_end. ray_end can be any point on the ray.
   // Returns a raycast result with the number of points of intersection
   // between the ray and the cube, an array of hit points (the lowest
   // index is the hit point nearest to ray_start), and an array of the ray
   // parameters leading to the hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCube_t & cube_C
   );

   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shape_t & shape_C
   );

   // The ray is formulated as:
   //    r(u) = ray_start + u * (ray_end - ray_start)
   // So that r(u=1) = ray_end. ray_end can be any point on the ray.
   // Returns a raycast result with the number of points of intersection
   // between the ray and the sphere, an array of hit points (the lowest index
   // is the hit point nearest to ray_start), and an array of the ray
   // parameters leading to the hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeSphere_t & sphere_C
   );

   // The ray is formulated as:
   //    r(u) = ray_start + u * (ray_end - ray_start)
   // So that r(u=1) = ray_end. ray_end can be any point on the ray.
   // Returns a raycast result with the number of points of intersection
   // between the ray and the cylinder, an array of hit points (the lowest
   // index is the hit point nearest to ray_start), and an array of the ray
   // parameters leading to the hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCylinder_t & cylinder_C
   );

   // The ray is formulated as:
   //    r(u) = ray_start + u * (ray_end - ray_start)
   // So that r(u=1) = ray_end. ray_end can be any point on the ray.
   // Returns a raycast result with the number of points of intersection
   // between the ray and the capsule, an array of hit points (the lowest
   // index is the hit point nearest to ray_start), and an array of the ray
   // parameters leading to the hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCapsule_t & capsule_C
   );

   // The ray is formulated as:
   //    r(u) = ray_start + u * (ray_end - ray_start)
   // So that r(u=1) = ray_end. ray_end can be any point on the ray.
   // Returns a raycast result with the number of points of intersection
   // between the ray and the cube, an array of hit points (the lowest
   // index is the hit point nearest to ray_start), and an array of the ray
   // parameters leading to the hit points.
   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCube_t & cube_C
   );

   geometry::types::raycastResult_t raycast(
      const Vector3 & ray_start,
      const Vector3 & ray_end,
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shape_t & shape_C
   );
}

#endif
