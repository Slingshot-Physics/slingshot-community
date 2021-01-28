#include "collision_contacts.hpp"

#include "attitudeutils.hpp"
#include "slingshot_types.hpp"
#include "mesh_ops.hpp"
#include "plane.hpp"
#include "polygon.hpp"
#include "segment.hpp"
#include "transform.hpp"
#include "triangle.hpp"

namespace oy
{

namespace collision
{
   geometry::types::polygon50_t clipCoplanarFeatures(
      const geometry::types::polyhedronFeature_t & feature_p,
      const geometry::types::polyhedronFeature_t & feature_q
   )
   {
      Matrix33 R_P_to_Q = makeVectorUp(feature_p.normal);

      Vector3 center_p = geometry::averageVertex(
         feature_p.shape.numVerts, feature_p.shape.verts
      );

      geometry::types::isometricTransform_t trans_W_to_C;
      trans_W_to_C.rotate = R_P_to_Q;
      trans_W_to_C.translate = -1.f * (R_P_to_Q * center_p);

      geometry::types::polygon50_t poly_p_C = feature_p.shape;
      geometry::types::polygon50_t poly_q_C = feature_q.shape;

      geometry::polygon::applyTransformation(trans_W_to_C, poly_p_C);
      geometry::polygon::applyTransformation(trans_W_to_C, poly_q_C);

      geometry::types::polygon50_t clipped_poly;
      geometry::polygon::convexIntersection(poly_p_C, poly_q_C, clipped_poly);

      // Create the transform to rotate the resulting clipped polygon back to
      // W coordinates from Clip coordinates.
      geometry::types::isometricTransform_t trans_C_to_W;
      trans_C_to_W.rotate = R_P_to_Q.transpose();
      trans_C_to_W.translate = center_p;

      geometry::polygon::applyTransformation(trans_C_to_W, clipped_poly);

      // Return the clipped polygon in W coordinates.
      return clipped_poly;
   }

   geometry::types::polygon4_t reducePolygon(
      const geometry::types::polygon50_t big_polygon
   )
   {
      float max_area = 0.f;
      unsigned int quad_vert_ids[4] = {0, 0, 0, 0};

      // Sooo ugly.
      for (unsigned int i = 0; i < big_polygon.numVerts; ++i)
      {
         for (unsigned int j = 0; j < big_polygon.numVerts; ++j)
         {
            if (j == i)
            {
               continue;
            }
            for (unsigned int k = 0; k < big_polygon.numVerts; ++k)
            {
               if (k == i || k == j)
               {
                  continue;
               }
               for (unsigned int m = 0; m < big_polygon.numVerts; ++m)
               {
                  if (m == i || m == j || m == k)
                  {
                     continue;
                  }
                  float temp_area = geometry::triangle::area(
                     big_polygon.verts[i],
                     big_polygon.verts[j],
                     big_polygon.verts[k]
                  );
                  temp_area += geometry::triangle::area(
                     big_polygon.verts[j],
                     big_polygon.verts[k],
                     big_polygon.verts[m]
                  );

                  if (temp_area > max_area)
                  {
                     max_area = temp_area;
                     quad_vert_ids[0] = i;
                     quad_vert_ids[1] = j;
                     quad_vert_ids[2] = k;
                     quad_vert_ids[3] = m;
                  }
               }
            }
         }
      }

      geometry::types::polygon4_t small_polygon;
      small_polygon.numVerts = 4;
      for (unsigned int i = 0; i < 4; ++i)
      {
         small_polygon.verts[i] = big_polygon.verts[quad_vert_ids[i]];
      }

      return small_polygon;
   }

   contactManifold_t referenceIncidentManifold(
      const geometry::types::polyhedronFeature_t & ref_feature_W,
      const geometry::types::polyhedronFeature_t & inc_feature_W,
      const geometry::types::plane_t ref_plane_W,
      const geometry::types::plane_t inc_plane_W,
      const geometry::types::plane_t col_plane_W
   )
   {
      // Project the reference polygon onto the incident plane through the
      // collision normal.
      geometry::types::polyhedronFeature_t ref_feature_inc_plane_W;
      ref_feature_inc_plane_W.shape = geometry::polygon::projectPolygonToPlane(
         ref_feature_W.shape, col_plane_W.normal, inc_plane_W
      );

      // Clip the projected features against each other, get the result in the
      // plane of the incident feature.
      geometry::types::polygon50_t inc_clipped_poly_W = clipCoplanarFeatures(
         inc_feature_W, ref_feature_inc_plane_W
      );

      geometry::types::polygon50_t inc_contact_points_W;
      unsigned int & numVerts = inc_contact_points_W.numVerts;
      numVerts = 0;

      // Keep points on the clipped result that are below the collision plane.
      for (unsigned int i = 0; i < inc_clipped_poly_W.numVerts; ++i)
      {
         const Vector3 & temp_vert = inc_clipped_poly_W.verts[i];
         float dot_product = (temp_vert - col_plane_W.point).dot(col_plane_W.normal);
         if (dot_product <= 0.f)
         {
            inc_contact_points_W.verts[numVerts] = temp_vert;
            numVerts += 1;
         }
      }

      // Reduce the manifold to a max of four vertices.
      geometry::types::polygon4_t small_inc_contact_points_W;
      if (inc_contact_points_W.numVerts > 4)
      {
         small_inc_contact_points_W = reducePolygon(inc_contact_points_W);
      }
      else
      {
         small_inc_contact_points_W.numVerts = inc_contact_points_W.numVerts;
         for (unsigned int i = 0; i < inc_contact_points_W.numVerts; ++i)
         {
            small_inc_contact_points_W.verts[i] = inc_contact_points_W.verts[i];
         }
      }

      contactManifold_t manifold;
      if (inc_contact_points_W.numVerts == 0)
      {
         manifold.numContacts = 0;
         return manifold;
      }

      geometry::types::polygon4_t small_ref_contact_points_W = \
         geometry::polygon::projectPolygonToPlane(
            small_inc_contact_points_W, col_plane_W.normal, ref_plane_W
         );

      manifold.numContacts = small_inc_contact_points_W.numVerts;
      for (unsigned int i = 0; i < manifold.numContacts; ++i)
      {
         manifold.incContacts_W[i] = small_inc_contact_points_W.verts[i];
         manifold.refContacts_W[i] = small_ref_contact_points_W.verts[i];
      }

      return manifold;
   }

}

}
