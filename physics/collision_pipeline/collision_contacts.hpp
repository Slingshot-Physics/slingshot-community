#ifndef COLLISION_CONTACTS_HEADER
#define COLLISION_CONTACTS_HEADER

#include "allocator.hpp"
#include "slingshot_types.hpp"
#include "geometry_types.hpp"
#include "shape_features.hpp"

namespace oy
{

namespace collision
{
   struct contactManifold_t
   {
      Vector3 refContacts_W[4];
      Vector3 incContacts_W[4];
      unsigned int numContacts;
   };

   // Clips the features 'feature_p' and 'feature_q' that are coplanar. Returns
   // the polygon result in the same plane as the arguments.
   geometry::types::polygon50_t clipCoplanarFeatures(
      const geometry::types::polyhedronFeature_t & feature_p,
      const geometry::types::polyhedronFeature_t & feature_q
   );

   // Finds the quadrilateral of maximal area in 'big_polygon' and returns the
   // result.
   geometry::types::polygon4_t reducePolygon(
      const geometry::types::polygon50_t big_polygon
   );

   // Reference feature - the feature whose normal is most aligned with the
   // normal of the collision plane.
   // Incident feature - the feature whose normal is less aligned with the
   // normal of the collision plane.
   // Reference plane - the plane passing through the contact point on the
   // reference body with a normal equal to the normal of the reference
   // feature.
   // Incident plane - the plane passing through the contact point on the
   // incident body with a normal equal to the normal of the incident feature.
   // Collision plane - the plane passing through the contact point on the
   // reference body with a normal equal to the collision normal.
   contactManifold_t referenceIncidentManifold(
      const geometry::types::polyhedronFeature_t & ref_feature_W,
      const geometry::types::polyhedronFeature_t & inc_feature_W,
      const geometry::types::plane_t ref_plane_W,
      const geometry::types::plane_t inc_plane_W,
      const geometry::types::plane_t col_plane_W
   );

   // Finds the features on two shapes whose surface normals are most parallel
   // to the collision normal and determines the contact manifold between those
   // two shapes.
   template <typename ShapeA_T, typename ShapeB_T, typename Transform_T>
   oy::types::collisionContactManifold_t calculateContactManifold(
      const ShapeA_T & shape_a_A,
      const ShapeB_T & shape_b_B,
      const Transform_T & trans_A_to_W,
      const Transform_T & trans_B_to_W,
      const oy::types::contactGeometry_t & contact_geometry
   )
   {
      geometry::types::plane_t query_plane_W;
      query_plane_W.normal = contact_geometry.contactNormal;
      query_plane_W.point = contact_geometry.bodyAContactPoint;

      geometry::types::polyhedronFeature_t feature_a_W = geometry::mostParallelFeature(
         trans_A_to_W, shape_a_A, query_plane_W
      );

      query_plane_W.normal *= -1.f;
      query_plane_W.point = contact_geometry.bodyBContactPoint;

      geometry::types::polyhedronFeature_t feature_b_W = geometry::mostParallelFeature(
         trans_B_to_W, shape_b_B, query_plane_W
      );

      float face_a_dot = feature_a_W.normal.dot(contact_geometry.contactNormal);
      float face_b_dot = feature_b_W.normal.dot(-1.f * contact_geometry.contactNormal);

      bool body_a_reference = false;

      geometry::types::plane_t ref_plane_W;
      geometry::types::plane_t inc_plane_W;
      geometry::types::plane_t col_plane_W;

      contactManifold_t manifold;

      if (face_a_dot > face_b_dot)
      {
         ref_plane_W.point = contact_geometry.bodyAContactPoint;
         ref_plane_W.normal = feature_a_W.normal;
         inc_plane_W.point = contact_geometry.bodyBContactPoint;
         inc_plane_W.normal = feature_b_W.normal;
         col_plane_W.point = contact_geometry.bodyAContactPoint;
         col_plane_W.normal = contact_geometry.contactNormal;
         manifold = referenceIncidentManifold(
            feature_a_W, feature_b_W, ref_plane_W, inc_plane_W, col_plane_W
         );
         body_a_reference = true;
      }
      else
      {
         ref_plane_W.point = contact_geometry.bodyBContactPoint;
         ref_plane_W.normal = feature_b_W.normal;
         inc_plane_W.point = contact_geometry.bodyAContactPoint;
         inc_plane_W.normal = feature_a_W.normal;
         col_plane_W.point = contact_geometry.bodyBContactPoint;
         col_plane_W.normal = -1.f * contact_geometry.contactNormal;
         manifold = referenceIncidentManifold(
            feature_b_W, feature_a_W, ref_plane_W, inc_plane_W, col_plane_W
         );
         body_a_reference = false;
      }

      oy::types::collisionContactManifold_t contacts;
      if (manifold.numContacts == 0)
      {
         contacts.bodyAContacts[0] = contact_geometry.bodyAContactPoint;
         contacts.bodyBContacts[0] = contact_geometry.bodyBContactPoint;
         contacts.numContacts = 1;
         return contacts;
      }

      contacts.numContacts = manifold.numContacts;
      if (body_a_reference)
      {
         for (unsigned int i = 0; i < contacts.numContacts; ++i)
         {
            contacts.bodyAContacts[i] = manifold.refContacts_W[i];
            contacts.bodyBContacts[i] = manifold.incContacts_W[i];
         }
      }
      else
      {
         for (unsigned int i = 0; i < contacts.numContacts; ++i)
         {
            contacts.bodyAContacts[i] = manifold.incContacts_W[i];
            contacts.bodyBContacts[i] = manifold.refContacts_W[i];
         }
      }

      return contacts;
   }

}

}

#endif
