#ifndef SHAPE_COLLISIONS_HEADER
#define SHAPE_COLLISIONS_HEADER

#include "geometry_types.hpp"
#include "matrix33.hpp"
#include "vector3.hpp"

namespace geometry
{

namespace collisions
{
   struct queryAxisMeta_t
   {
      bool separated;
      float R_a;
      float R_b;
      float l_dot_t;

      float normalizedDistance(void)
      {
         return (R_a + R_b) / l_dot_t;
      }
   };

   struct cubeCubeCollisionType_t
   {
      enum {
         FACE_A,
         FACE_B,
         EDGES
      } feature;
      Vector3 faceANormal_W;
      Vector3 faceBNormal_W;
      unsigned int edgeAId;
      unsigned int edgeBId;
   };

   // Calculates the SAT function for a given test axis, support points, and
   // center of geometry separation vector.
   // The support points support_a_P and support_b_P should be relative to each
   // shape's geometric center.
   queryAxisMeta_t separationAxis(
      const Vector3 & support_a_P,
      const Vector3 & support_b_P,
      const Vector3 & a_to_b_P,
      const Vector3 & test_axis_P
   );

   geometry::types::satResult_t cubeCube(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeCube_t & cube_b_B
   );

   geometry::types::satResult_t cubeSphere(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeSphere_t & sphere_b_B
   );

   geometry::types::satResult_t sphereSphere(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & shape_a_A,
      const geometry::types::shapeSphere_t & shape_b_B
   );

   geometry::types::satResult_t sphereCapsule(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & shape_a_A,
      const geometry::types::shapeCapsule_t & shape_b_B
   );

   geometry::types::satResult_t sphereCylinder(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & shape_a_A,
      const geometry::types::shapeCylinder_t & shape_b_B
   );

   geometry::types::satResult_t capsuleCapsule(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCapsule_t & capsule_a_A,
      const geometry::types::shapeCapsule_t & capsule_b_B
   );

   inline geometry::types::labeledVertex_t segmentSupport(
      const Vector3 & dir_B,
      const Vector3 & a_B,
      const Vector3 & b_B
   )
   {
      geometry::types::labeledVertex_t result;
      const Vector3 center = (a_B + b_B) / 2.f;
      if ((a_B - center).dot(dir_B) > (b_B - center).dot(dir_B))
      {
         result.vert = a_B;
         result.vertId = 0;
      }
      else
      {
         result.vert = b_B;
         result.vertId = 1;
      }

      return result;
   }

   // Determines if a cube and a capsule are separated. The SAT result will
   // have collision = true if they're colliding, false otherwise.
   // The SAT result will contain a separating axis for collision resolution
   // and one or two pseudo-contact points for contact manifold calculation.
   // Pseudo-contact points are guaranteed to be on the shape feature
   // containing the contact manifold.
   geometry::types::satResult_t cubeCapsule(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeCapsule_t & capsule_b_B
   );

   // Assumes that both AABB's are in the same coordinate frame.
   bool aabbAabb(
      const geometry::types::aabb_t & aabb_a,
      const geometry::types::aabb_t & aabb_b
   );

   inline geometry::types::satResult_t sat(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & shape_a_A,
      const geometry::types::shapeSphere_t & shape_b_B
   )
   {
      return sphereSphere(
         trans_A_to_W,
         trans_B_to_W,
         shape_a_A,
         shape_b_B
      );
   }

   inline geometry::types::satResult_t sat(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & shape_a_A,
      const geometry::types::shapeCapsule_t & shape_b_B
   )
   {
      return sphereCapsule(
         trans_A_to_W,
         trans_B_to_W,
         shape_a_A,
         shape_b_B
      );
   }

   inline geometry::types::satResult_t sat(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & shape_a_A,
      const geometry::types::shapeCylinder_t & shape_b_B
   )
   {
      return sphereCylinder(
         trans_A_to_W,
         trans_B_to_W,
         shape_a_A,
         shape_b_B
      );
   }

   inline geometry::types::satResult_t sat(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCapsule_t & shape_a_A,
      const geometry::types::shapeCapsule_t & shape_b_B
   )
   {
      return capsuleCapsule(
         trans_A_to_W,
         trans_B_to_W,
         shape_a_A,
         shape_b_B
      );
   }

   inline geometry::types::satResult_t sat(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeCube_t & cube_b_B
   )
   {
      return cubeCube(
         trans_A_to_W,
         trans_B_to_W,
         cube_a_A,
         cube_b_B
      );
   }

   inline geometry::types::satResult_t sat(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeSphere_t & sphere_b_B
   )
   {
      return cubeSphere(
         trans_A_to_W,
         trans_B_to_W,
         cube_a_A,
         sphere_b_B
      );
   }

   inline geometry::types::satResult_t sat(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeCapsule_t & capsule_b_B
   )
   {
      return cubeCapsule(
         trans_A_to_W,
         trans_B_to_W,
         cube_a_A,
         capsule_b_B
      );
   }
}

}

#endif
