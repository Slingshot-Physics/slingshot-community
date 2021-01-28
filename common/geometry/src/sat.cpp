#include "sat.hpp"

#include "plane.hpp"
#include "segment.hpp"
#include "support_functions.hpp"
#include "transform.hpp"

#include <algorithm>

namespace geometry
{

namespace collisions
{
   queryAxisMeta_t separationAxis(
      const Vector3 & support_a_P,
      const Vector3 & support_b_P,
      const Vector3 & a_to_b_P,
      const Vector3 & test_axis_P
   )
   {
      queryAxisMeta_t result;
      result.R_a = std::fabs(support_a_P.dot(test_axis_P));
      result.R_b = std::fabs(support_b_P.dot(test_axis_P));
      result.l_dot_t = std::fabs(a_to_b_P.dot(test_axis_P));
      result.separated = (result.R_a + result.R_b) < result.l_dot_t;

      return result;
   }

   // This is a truncated SAT with an assumption of symmetry on the input
   // shapes. The axes tested are:
   //    - All face normals of A
   //    - All face normals of B
   //    - The vectors resulting from cross products of all unique edge vectors
   //      between cube A and cube B.
   // Edge-edge test axes can be the zero vector (or very close to the zero
   // vector) if the two edge vectors are roughly parallel to each other. An
   // unhandled zero-length test axis will make the algorithm say the bodies
   // are not colliding.
   // This implementation handles parallel-ish edge vectors by:
   //    - Trying to calculate a vector that's orthogonal to edge_a and edge_b,
   //      call it m
   //    - If m.magnitude() < 1e-7f
   //      - Mark the bodies as separated *only* if some other test axis has
   //        not produced a collision. This is the key: if a collision is
   //        detected in an earlier stage, then we're still trying to find an
   //        axis that produces a minimal normalized overlap distance. Ignoring
   //        the degenerate edges rather than letting them say 'no collision'
   //        is better in this case.
   //    - Else
   //      - The test axis becomes edge_a x m
   geometry::types::satResult_t cubeCube(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeCube_t & cube_b_B
   )
   {
      geometry::types::satResult_t result;

      float best_normalized_overlap = __FLT_MAX__;
      Vector3 best_axis_W;

      cubeCubeCollisionType_t collision_type;

      const Vector3 d_ab_W = trans_B_to_W.translate - trans_A_to_W.translate;

      const geometry::ShapeSupport<geometry::types::shapeCube_t> support_a(
         trans_A_to_W, cube_a_A
      );

      const geometry::ShapeSupport<geometry::types::shapeCube_t> support_b(
         trans_B_to_W, cube_b_B
      );

      // For transforming the face normal in collider coordinates to a face
      // normal in world coordinates.
      const Matrix33 face_a_normal_trans = trans_A_to_W.rotate;

      // Check the face normals of cube A for separation
      for (unsigned int i = 0; i < 3; ++i)
      {
         Vector3 face_a_normal_A;
         face_a_normal_A[i] = 1.f;
         Vector3 test_axis_W = face_a_normal_trans * face_a_normal_A;

         geometry::types::labeledVertex_t labeled_support_a_W = \
            support_a.unboundSupportBodyDir(face_a_normal_A);

         geometry::types::labeledVertex_t labeled_support_b_W = \
            support_b.unboundSupportWorldDir(test_axis_W);

         queryAxisMeta_t query_result = separationAxis(
            labeled_support_a_W.vert,
            labeled_support_b_W.vert,
            d_ab_W,
            test_axis_W
         );

         if (query_result.separated)
         {
            result.collision = false;
            return result;
         }

         result.collision = true;
         if (query_result.normalizedDistance() < best_normalized_overlap)
         {
            collision_type.feature = cubeCubeCollisionType_t::FACE_A;
            collision_type.faceANormal_W = test_axis_W;
            best_normalized_overlap = query_result.normalizedDistance();
            best_axis_W = test_axis_W;
         }
      }

      const Matrix33 face_b_normal_trans = trans_B_to_W.rotate;

      // Check the face normals of cube B for separation
      for (unsigned int i = 0; i < 3; ++i)
      {
         Vector3 face_b_normal_B;
         face_b_normal_B[i] = 1.f;
         Vector3 test_axis_W = face_b_normal_trans * face_b_normal_B;

         geometry::types::labeledVertex_t labeled_support_a_W = \
            support_a.unboundSupportWorldDir(test_axis_W);

         geometry::types::labeledVertex_t labeled_support_b_W = \
            support_b.unboundSupportBodyDir(face_b_normal_B);

         queryAxisMeta_t query_result = separationAxis(
            labeled_support_a_W.vert,
            labeled_support_b_W.vert,
            d_ab_W,
            test_axis_W
         );

         if (query_result.separated)
         {
            result.collision = false;
            return result;
         }

         result.collision = true;
         if (query_result.normalizedDistance() < best_normalized_overlap)
         {
            collision_type.feature = cubeCubeCollisionType_t::FACE_B;
            collision_type.faceBNormal_W = test_axis_W;
            best_normalized_overlap = query_result.normalizedDistance();
            best_axis_W = test_axis_W;
         }
      }

      // One of the cube corners in its body frame.
      const Vector3 base_vert_M(-1.f, -1.f, -1.f);

      // Check the cross product of each edge of cube A with each edge of
      // cube B for separation.
      for (unsigned int i = 0; i < 3; ++i)
      {
         Vector3 edge_a_A;
         edge_a_A[i] = 1.f;
         Vector3 edge_a_W = geometry::transform::forwardUnbound(
            trans_A_to_W, edge_a_A
         );

         for (unsigned int j = 0; j < 3; ++j)
         {
            Vector3 edge_b_B;
            edge_b_B[j] = 1.f;
            Vector3 edge_b_W = geometry::transform::forwardUnbound(
               trans_B_to_W, edge_b_B
            );

            // dir_vert_b_B is a vertex on cube B that's in the edge_b_B
            // direction from base_vert_M.
            Vector3 dir_vert_b_B(-1.f, -1.f, -1.f);
            dir_vert_b_B[j] = 1.f;

            Vector3 test_axis_W = edge_a_W.crossProduct(edge_b_W);
            // The edges are pretty dang parallel, so try to work around this.
            if (test_axis_W.magnitude() < 1e-6f)
            {
               // Get the negative-most corner of cube A in world coordinates
               Vector3 base_vert_a_W = geometry::transform::forwardBound(
                  trans_A_to_W, base_vert_M
               );

               // Get the vertex on cube B that's in the query edge direction
               Vector3 dir_vert_b_W = geometry::transform::forwardBound(
                  trans_B_to_W, dir_vert_b_B
               );

               edge_a_W = dir_vert_b_W - base_vert_a_W;

               const Vector3 n = edge_a_W.crossProduct(edge_b_W);
               test_axis_W = n.crossProduct(edge_b_W);
            }

            geometry::types::labeledVertex_t labeled_support_a_W = \
               support_a.unboundSupportWorldDir(test_axis_W);

            geometry::types::labeledVertex_t labeled_support_b_W = \
               support_b.unboundSupportWorldDir(test_axis_W);

            queryAxisMeta_t query_result = separationAxis(
               labeled_support_a_W.vert,
               labeled_support_b_W.vert,
               d_ab_W,
               test_axis_W
            );

            if (query_result.separated)
            {
               result.collision = false;
               return result;
            }

            result.collision = true;
            if (query_result.normalizedDistance() < best_normalized_overlap)
            {
               collision_type.feature = cubeCubeCollisionType_t::EDGES;
               collision_type.edgeAId = i;
               collision_type.edgeBId = j;
               best_normalized_overlap = query_result.normalizedDistance();
               best_axis_W = test_axis_W;
            }
         }
      }

      if (best_axis_W.dot(d_ab_W) < 0.f)
      {
         // Make sure the collision normal points from A to B
         best_axis_W *= -1.f;
      }

      best_axis_W.Normalize();

      // Find some points on the supporting features that can be treated as
      // faux contact points for the contact manifold calculator.

      // Used to transform the query direction for calculating support vertices
      // on cube A.
      const Matrix33 support_dir_mat_A = supportDirectionTransform(trans_A_to_W);
      const Matrix33 support_dir_mat_B = supportDirectionTransform(trans_B_to_W);

      Vector3 support_a_A = geometry::supportMapping(
         support_dir_mat_A * best_axis_W, cube_a_A
      ).vert;

      Vector3 support_b_B = geometry::supportMapping(
         -1.f * support_dir_mat_B * best_axis_W, cube_b_B
      ).vert;

      Vector3 support_a_W = geometry::transform::forwardBound(
         trans_A_to_W, support_a_A
      );

      Vector3 support_b_W = geometry::transform::forwardBound(
         trans_B_to_W, support_b_B
      );

      result.numDeepestPointPairs = 1;
      switch(collision_type.feature)
      {
         case cubeCubeCollisionType_t::FACE_A:
         {
            result.deepestPointsB[0] = support_b_W;

            result.deepestPointsA[0] = geometry::plane::closestPointToPoint(
               collision_type.faceANormal_W, support_a_W, support_b_W
            );
            break;
         }
         case cubeCubeCollisionType_t::FACE_B:
         {
            result.deepestPointsA[0] = support_a_W;

            result.deepestPointsB[0] = geometry::plane::closestPointToPoint(
               collision_type.faceBNormal_W, support_b_W, support_a_W
            );
            break;
         }
         case cubeCubeCollisionType_t::EDGES:
         {
            // Get the vertices on each edge by treating each body frame
            // support point as a support direction and multiplying it by
            // negative 1.
            Vector3 dir_vert_a_A(support_a_A);
            dir_vert_a_A[collision_type.edgeAId] *= -1.f;

            Vector3 dir_vert_a_W = geometry::transform::forwardBound(
               trans_A_to_W, dir_vert_a_A
            );

            Vector3 dir_vert_b_B(support_b_B);
            dir_vert_b_B[collision_type.edgeBId] *= -1.f;

            Vector3 dir_vert_b_W = geometry::transform::forwardBound(
               trans_B_to_W, dir_vert_b_B
            );

            const auto segment_closest_points = \
               geometry::segment::closestPointsToSegment(
                  support_a_W, dir_vert_a_W, support_b_W, dir_vert_b_W, 1e-5f
               );

            result.deepestPointsA[0] = segment_closest_points.segmentPoints[0];
            result.deepestPointsB[0] = segment_closest_points.otherPoints[0];

            break;
         }
      }

      result.collision = true;
      result.contactNormal = best_axis_W;
      return result;
   }

   // This function performs most of its calculations in the cube's body frame.
   // Collision is detected by determining if the sphere's cube-clamped center
   // is within one sphere radius from the sphere's actual center.
   // The contact normal is determined by either the vector from the closest
   // point on the cube to the sphere, or a vector from one of the cube faces.
   geometry::types::satResult_t cubeSphere(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeSphere_t & sphere_b_B
   )
   {
      const Vector3 sphere_center_A = geometry::transform::inverseBound(
         trans_A_to_W, trans_B_to_W.translate
      );

      // Clamp the sphere's center to a point inside the cube in the cube's
      // body coordinates.
      const Vector3 closest_point_in_cube_A(
         std::max(
            std::min(sphere_center_A[0], cube_a_A.length / 2.f),
            -1.f * cube_a_A.length / 2.f
         ),
         std::max(
            std::min(sphere_center_A[1], cube_a_A.width / 2.f),
            -1.f * cube_a_A.width / 2.f
         ),
         std::max(
            std::min(sphere_center_A[2], cube_a_A.height / 2.f),
            -1.f * cube_a_A.height / 2.f
         )
      );

      const float closest_points_m2 = (
         sphere_center_A - closest_point_in_cube_A
      ).magnitudeSquared();

      // The sphere and the cube are colliding if the distance between the
      // sphere's center point and the sphere's center clamped to the cube
      // boundaries is less than the sphere's radius.
      geometry::types::satResult_t result;
      result.collision = closest_points_m2 < (sphere_b_B.radius * sphere_b_B.radius);
      result.numDeepestPointPairs = 1;

      Vector3 closest_point_on_cube_A(closest_point_in_cube_A);
      Vector3 contact_normal_A(sphere_center_A - closest_point_on_cube_A);

      // If the sphere's center is inside the cube, find the point on the
      // surface of the cube that's closest to the sphere's center and
      // calculate the collision normal from the closest face to the sphere's
      // center.
      if (closest_points_m2 <= 1e-6f)
      {
         const float cube_dimensions_A[3] = {
            cube_a_A.length, cube_a_A.width, cube_a_A.height
         };

         // Find the face of the cube that generates the collision normal and
         // that has the contact point.
         int smallest_index = 0;
         float closest_face_dist = (
            cube_dimensions_A[0] / 2.f - closest_point_in_cube_A[0]
         ) * (
            (closest_point_in_cube_A[0] >= 0.f) ? 1.f : -1.f
         );

         for (int i = 1; i < 3; ++i)
         {
            float temp_face_dist = (
               cube_dimensions_A[i] / 2.f - closest_point_in_cube_A[i]
            ) * (
               (closest_point_in_cube_A[i] >= 0.f) ? 1.f : -1.f
            );

            if (temp_face_dist < closest_face_dist)
            {
               smallest_index = i;
               closest_face_dist = temp_face_dist;
            }
         }

         const float sign_val = (
            closest_point_in_cube_A[smallest_index] >= 0.f
         ) ? 1.f : -1.f;

         closest_point_on_cube_A[smallest_index] = (
            sign_val * cube_dimensions_A[smallest_index] / 2.f
         );

         contact_normal_A.Initialize(0.f, 0.f, 0.f);
         contact_normal_A[smallest_index] = sign_val;
      }

      result.contactNormal = geometry::transform::forwardUnbound(
         trans_A_to_W, contact_normal_A
      );
      result.contactNormal.Normalize();

      result.deepestPointsA[0] = geometry::transform::forwardBound(
         trans_A_to_W, closest_point_on_cube_A
      );

      const Vector3 closest_point_on_sphere_A = sphere_b_B.radius * (
         closest_point_on_cube_A - sphere_center_A
      ).unitVector() + sphere_center_A;

      result.deepestPointsB[0] = geometry::transform::forwardBound(
         trans_A_to_W, closest_point_on_sphere_A
      );

      return result;
   }

   // The bulk of this computation is performed in the cube's body frame.
   geometry::types::satResult_t cubeCapsule(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCube_t & cube_a_A,
      const geometry::types::shapeCapsule_t & capsule_b_B
   )
   {
      geometry::types::satResult_t result;

      // The logic flow of this function is:
      // - Find closest point pairs between the capsule's symmetry segment and
      //   the cube
      // - If the distance between a closest point pair is greater than the
      //   capsule radius, there is no collision
      // - If the distance between a closest point pair is less than the
      //   capsule radius and greater than zero:
      //   - A collision has occurred
      //   - Set the separation axis as the vector between the first closest
      //     point pair
      //   - Set the contact points as the closest point pairs
      // - If the distance between a closest point pair is pretty close to zero
      //   - A collision has occurred
      //   - Use the SAT rules for convex polyhedra to find a separating axis
      //     that minimizes the SAT function
      //   - Generate pseudo contact points using support points on each body
      //     in the direction of the support axis

      const Vector3 capsule_points_A[2] = {
         geometry::transform::inverseBound(
            trans_A_to_W,
            geometry::transform::forwardBound(
               trans_B_to_W, {0.f, 0.f, capsule_b_B.height / 2.f}
            )
         ),
         geometry::transform::inverseBound(
            trans_A_to_W,
            geometry::transform::forwardBound(
               trans_B_to_W, {0.f, 0.f, -1.f * capsule_b_B.height / 2.f}
            )
         )
      };

      geometry::types::aabb_t cube_aabb_A;
      cube_aabb_A.vertMax.Initialize(
         cube_a_A.length / 2.f, cube_a_A.width / 2.f, cube_a_A.height / 2.f
      );
      cube_aabb_A.vertMin = -1.f * cube_aabb_A.vertMax;

      // Vector between the centers of the cube and the segment in the cube's
      // body frame
      const Vector3 d_ab_A = (capsule_points_A[0] + capsule_points_A[1]) / 2.f;

      // Get the distance between the closest points on the capsule segment
      // and the cube AABB.
      const auto segment_aabb_closest_points = geometry::segment::closestPointsToAabb(
         capsule_points_A[0], capsule_points_A[1], cube_aabb_A
      );

      const float segment_aabb_distance = (
         segment_aabb_closest_points.segmentPoints[0] - segment_aabb_closest_points.otherPoints[0]
      ).magnitude();

      // No intersection between cube and capsule
      if (segment_aabb_distance > capsule_b_B.radius)
      {
         result.collision = false;
         return result;
      }

      // Intersection between cube and capsule but capsule segment doesn't
      // intersect cube
      if (
         (segment_aabb_distance <= capsule_b_B.radius) &&
         (segment_aabb_distance > 1e-7f)
      )
      {
         result.collision = true;
         Vector3 contact_normal_A = (
            segment_aabb_closest_points.segmentPoints[0] - segment_aabb_closest_points.otherPoints[0]
         );

         // Make sure the contact normal points from shape A to shape B
         if (contact_normal_A.dot(d_ab_A) < 0.f)
         {
            contact_normal_A *= -1.f;
         }

         result.contactNormal = geometry::transform::forwardUnbound(
            trans_A_to_W, contact_normal_A
         );
         result.contactNormal.Normalize();
         result.numDeepestPointPairs = 1;
         result.deepestPointsA[0] = geometry::transform::forwardBound(
            trans_A_to_W, segment_aabb_closest_points.otherPoints[0]
         );

         // Have to expand the contact point on the segment to a point on the
         // surface of the capsule
         result.deepestPointsB[0] = geometry::transform::forwardBound(
            trans_A_to_W,
            segment_aabb_closest_points.segmentPoints[0] - capsule_b_B.radius * contact_normal_A.unitVector()
         );

         return result;
      }

      // The cube and capsule collide and the capsule's symmetry segment
      // intersects the cube. Use SAT between the cube and the segment to find
      // a separating axis with minimal overlap.
      // Separating axes can be:
      //    - faces of the cube
      //    - the cross-product of the segment vector and all cube edges
      // With all the same caveats of edge cross-products as cube-cube.

      float best_normalized_overlap = __FLT_MAX__;
      Vector3 best_axis_A;

      const Vector3 segment_edge_A = capsule_points_A[1] - capsule_points_A[0];
      const Vector3 segment_center_A = (capsule_points_A[1] + capsule_points_A[0]) / 2.f;

      // Check the segment vector for separation
      {
         geometry::types::labeledVertex_t labeled_support_a_A = \
            geometry::supportMapping(segment_edge_A, cube_a_A);

         geometry::types::labeledVertex_t labeled_support_b_A = {
            1, capsule_points_A[1]
         };

         queryAxisMeta_t query_result = separationAxis(
            labeled_support_a_A.vert,
            labeled_support_b_A.vert - segment_center_A,
            d_ab_A,
            segment_edge_A
         );

         if (query_result.normalizedDistance() < best_normalized_overlap)
         {
            best_normalized_overlap = query_result.normalizedDistance();
            best_axis_A = segment_edge_A;
         }
      }

      // Check the cube's face normals for separation
      for (unsigned int i = 0; i < 3; ++i)
      {
         Vector3 face_normal_A;
         face_normal_A[i] = 1.f;

         geometry::types::labeledVertex_t labeled_support_a_A = \
            geometry::supportMapping(face_normal_A, cube_a_A);

         geometry::types::labeledVertex_t labeled_support_b_A = \
            segmentSupport(
               -1.f * face_normal_A, capsule_points_A[0], capsule_points_A[1]
            );

         queryAxisMeta_t query_result = separationAxis(
            labeled_support_a_A.vert,
            labeled_support_b_A.vert - segment_center_A,
            d_ab_A,
            face_normal_A
         );

         if (query_result.normalizedDistance() < best_normalized_overlap)
         {
            best_normalized_overlap = query_result.normalizedDistance();
            best_axis_A = face_normal_A;
         }
      }

      const float cube_dims[3] = {
         cube_a_A.length / 2.f, cube_a_A.width / 2.f, cube_a_A.height / 2.f
      };

      // Check the cross-products of the segment edge with the cube edges for
      // separation
      for (unsigned int i = 0; i < 3; ++i)
      {
         Vector3 cube_edge_A;
         cube_edge_A[i] = 1.f;
         const Vector3 test_axis_A = segment_edge_A.crossProduct(cube_edge_A);

         if (test_axis_A.magnitude() < 1e-6f)
         {
            // Get the vertices on the most negative face whose normal is
            // +/- axis[i]
            Vector3 cube_vertices_A[4];
            for (unsigned int j = 0; j < 4; ++j)
            {
               cube_vertices_A[j][(i + 0) % 3] = cube_dims[(i + 0) % 3] * (
                  -1.f
               );
               cube_vertices_A[j][(i + 1) % 3] = cube_dims[(i + 1) % 3] * (
                  (j == 0 || j == 3) ? 1.f : -1.f
               );
               cube_vertices_A[j][(i + 2) % 3] = cube_dims[(i + 2) % 3] * (
                  (j == 1 || j == 2) ? 1.f : -1.f
               );
            }

            // Find the face vertex closest to one of the capsule points
            float closest_vertex_dist = __FLT_MAX__;
            Vector3 closest_vertex_A;

            for (unsigned int j = 0; j < 4; ++j)
            {
               const float temp_dist = (cube_vertices_A[j] - capsule_points_A[0]).magnitudeSquared();
               if (temp_dist < closest_vertex_dist)
               {
                  closest_vertex_dist = temp_dist;
                  closest_vertex_A = cube_vertices_A[j];
               }
            }

            // Construct the potential separating axis
            // (I wasn't paying close attention to this - it might be wrong)
            const Vector3 faux_edge_A = closest_vertex_A - capsule_points_A[0];
            const Vector3 n = faux_edge_A.crossProduct(cube_edge_A);
            test_axis_A = n.crossProduct(cube_edge_A);

            // If the test axis still has zero magnitude it means that the
            // capsule axis is completely colinear with one of the cube edges,
            // which means that this axis can be skipped entirely
            if (test_axis_A.magnitude() == 0)
            {
               continue;
            }
         }

         geometry::types::labeledVertex_t labeled_support_a_A = \
            geometry::supportMapping(test_axis_A, cube_a_A);

         geometry::types::labeledVertex_t labeled_support_b_A = \
            segmentSupport(
               -1.f * test_axis_A, capsule_points_A[0], capsule_points_A[1]
            );

         queryAxisMeta_t query_result = separationAxis(
            labeled_support_a_A.vert,
            labeled_support_b_A.vert - segment_center_A,
            d_ab_A,
            test_axis_A
         );

         if (query_result.normalizedDistance() < best_normalized_overlap)
         {
            best_normalized_overlap = query_result.normalizedDistance();
            best_axis_A = test_axis_A;
         }
      }

      result.collision = true;

      // Make sure the contact normal points from shape A to shape B
      if (best_axis_A.dot(d_ab_A) < 0.f)
      {
         best_axis_A *= -1.f;
      }

      result.contactNormal = geometry::transform::forwardUnbound(
         trans_A_to_W, best_axis_A
      );
      result.contactNormal.Normalize();

      // Calculate pseudo-contact points as the support points along the
      // separating axis
      result.numDeepestPointPairs = 1;
      result.deepestPointsA[0] = geometry::transform::forwardBound(
         trans_A_to_W, geometry::supportMapping(best_axis_A, cube_a_A).vert
      );

      const Vector3 capsule_contact_A = segmentSupport(
         -1.f * best_axis_A, capsule_points_A[0], capsule_points_A[1]
      ).vert - capsule_b_B.radius * best_axis_A.unitVector();

      result.deepestPointsB[0] = geometry::transform::forwardBound(
         trans_A_to_W, capsule_contact_A
      );

      return result;
   }

   geometry::types::satResult_t sphereSphere(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & sphere_a_A,
      const geometry::types::shapeSphere_t & sphere_b_B
   )
   {
      geometry::types::satResult_t result;
      Vector3 sep_axis_W = trans_B_to_W.translate - trans_A_to_W.translate;
      Vector3 normal_W = sep_axis_W.unitVector();
      result.contactNormal = normal_W;

      result.deepestPointsA[0] = trans_A_to_W.translate + normal_W * sphere_a_A.radius;
      result.deepestPointsB[0] = trans_B_to_W.translate - normal_W * sphere_b_B.radius;
      result.numDeepestPointPairs = 1;
      result.collision = (
         sep_axis_W.magnitude() <= (sphere_a_A.radius + sphere_b_B.radius)
      );

      return result;
   }

   geometry::types::satResult_t sphereCapsule(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & sphere_a_A,
      const geometry::types::shapeCapsule_t & capsule_b_B
   )
   {
      Vector3 cap_verts_b_B[2] = {
         Vector3(0.f, 0.f, capsule_b_B.height / 2),
         Vector3(0.f, 0.f, -capsule_b_B.height / 2)
      };

      Vector3 cap_verts_b_W[2] = {
         geometry::transform::forwardBound(trans_B_to_W, cap_verts_b_B[0]),
         geometry::transform::forwardBound(trans_B_to_W, cap_verts_b_B[1]),
      };

      Vector3 closest_seg_point_W = geometry::segment::closestPointToPoint(
         cap_verts_b_W[0], cap_verts_b_W[1], trans_A_to_W.translate
      ).point;

      geometry::types::satResult_t result;
      result.numDeepestPointPairs = 1;

      Vector3 collision_axis_W = closest_seg_point_W - trans_A_to_W.translate;
      result.contactNormal = collision_axis_W.unitVector();
      const Vector3 & normal_W = result.contactNormal;

      result.deepestPointsA[0] = sphere_a_A.radius * normal_W + trans_A_to_W.translate;
      result.deepestPointsB[0] = -1.f * capsule_b_B.radius * normal_W + closest_seg_point_W;
      result.collision = (
         collision_axis_W.magnitude() <= (sphere_a_A.radius + capsule_b_B.radius)
      );

      return result;
   }

   geometry::types::satResult_t sphereCylinder(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeSphere_t & shape_a_A,
      const geometry::types::shapeCylinder_t & shape_b_B
   )
   {
      const Vector3 sphere_center_B = geometry::transform::inverseBound(
         trans_B_to_W, trans_A_to_W.translate
      );

      const float sphere_xy_dist = std::sqrt(
         sphere_center_B[0] * sphere_center_B[0] + sphere_center_B[1] * sphere_center_B[1]
      );
      const float radial_scale = shape_b_B.radius / std::max(sphere_xy_dist, 1e-7f);

      const Vector3 closest_point_tube_B = {
         sphere_center_B[0] * radial_scale,
         sphere_center_B[1] * radial_scale,
         std::max(
            std::min(
               sphere_center_B[2], shape_b_B.height / 2.f
            ),
            -1.f * shape_b_B.height / 2.f
         )
      };

      const Vector3 closest_point_cap_B = {
         sphere_center_B[0] * (
            (sphere_xy_dist < shape_b_B.radius) ? 1.f : radial_scale
         ),
         sphere_center_B[1] * (
            (sphere_xy_dist < shape_b_B.radius) ? 1.f : radial_scale
         ),
         (shape_b_B.height / 2.f) * (sphere_center_B[2] >= 0.f ? 1.f : -1.f)
      };

      const float tube_dist2 = (sphere_center_B - closest_point_tube_B).magnitudeSquared();
      const float cap_dist2 = (sphere_center_B - closest_point_cap_B).magnitudeSquared();

      const Vector3 closest_point_cylinder_B = (
         (
            (tube_dist2 <= cap_dist2) && (sphere_xy_dist > 1e-7f)
         )
         ? closest_point_tube_B
         : closest_point_cap_B
      );

      const bool sphere_center_inside_cylinder = (
         (sphere_xy_dist <= shape_b_B.radius) &&
         (std::abs(sphere_center_B[2]) <= shape_b_B.height / 2.f)
      );

      geometry::types::satResult_t result;
      result.collision = (
         sphere_center_inside_cylinder ||
         (
            std::min(tube_dist2, cap_dist2) <= (shape_a_A.radius * shape_a_A.radius)
         )
      );

      if (!result.collision)
      {
         return result;
      }

      const float contact_normal_sign = sphere_center_inside_cylinder ? -1.f : 1.f;

      result.contactNormal = contact_normal_sign * geometry::transform::forwardUnbound(
         trans_B_to_W, closest_point_cylinder_B - sphere_center_B
      );
      result.contactNormal.Normalize();

      // If the center of the sphere is on the surface of the cylinder, then
      // use the cylinder's surface normal as the contact normal.
      if (result.contactNormal.magnitudeSquared() < 1e-7f)
      {
         Vector3 contact_normal_B;
         if (std::abs(closest_point_cylinder_B[2]) >= (shape_b_B.height / 2.f - 1e-7f))
         {
            contact_normal_B.Initialize(
               0.f,
               0.f,
               (shape_b_B.height / 2.f) * (
                  (closest_point_cylinder_B[2] >= 0.f)
                  ? 1.f
                  : -1.f
               )
            );
         }
         else
         {
            contact_normal_B = closest_point_cylinder_B;
            contact_normal_B[2] = 0.f;
         }

         result.contactNormal = -1.f * geometry::transform::forwardUnbound(
            trans_B_to_W, contact_normal_B
         );

         result.contactNormal.Normalize();
      }

      result.numDeepestPointPairs = 1;

      result.deepestPointsA[0] = (
         trans_A_to_W.translate + result.contactNormal * shape_a_A.radius
      );

      result.deepestPointsB[0] = geometry::transform::forwardBound(
         trans_B_to_W, closest_point_cylinder_B
      );

      return result;
   }

   geometry::types::satResult_t capsuleCapsule(
      const geometry::types::isometricTransform_t & trans_A_to_W,
      const geometry::types::isometricTransform_t & trans_B_to_W,
      const geometry::types::shapeCapsule_t & capsule_a_A,
      const geometry::types::shapeCapsule_t & capsule_b_B
   )
   {
      geometry::types::satResult_t result;

      Vector3 cap_verts_a_A[2] = {
         Vector3(0.f, 0.f, capsule_a_A.height / 2),
         Vector3(0.f, 0.f, -capsule_a_A.height / 2)
      };
      Vector3 cap_verts_b_B[2] = {
         Vector3(0.f, 0.f, capsule_b_B.height / 2),
         Vector3(0.f, 0.f, -capsule_b_B.height / 2)
      };

      Vector3 cap_verts_a_W[2] = {
         geometry::transform::forwardBound(trans_A_to_W, cap_verts_a_A[0]),
         geometry::transform::forwardBound(trans_A_to_W, cap_verts_a_A[1])
      };
      Vector3 cap_verts_b_W[2] = {
         geometry::transform::forwardBound(trans_B_to_W, cap_verts_b_B[0]),
         geometry::transform::forwardBound(trans_B_to_W, cap_verts_b_B[1])
      };

      // The closest points on both bodies
      const auto closest_segment_points = geometry::segment::closestPointsToSegment(
         cap_verts_a_W[0],
         cap_verts_a_W[1],
         cap_verts_b_W[0],
         cap_verts_b_W[1],
         1e-3f
      );

      const Vector3 & p = closest_segment_points.segmentPoints[0];
      const Vector3 & q = closest_segment_points.segmentPoints[1];
      const Vector3 & r = closest_segment_points.otherPoints[0];
      const Vector3 & s = closest_segment_points.otherPoints[1];

      const Vector3 contact_vector = r - p;
      result.contactNormal = contact_vector.unitVector();
      result.numDeepestPointPairs = closest_segment_points.numPairs;
      result.collision = (contact_vector.magnitude() <= (capsule_a_A.radius + capsule_b_B.radius));

      if (closest_segment_points.numPairs == 1)
      {
         result.deepestPointsA[0] = capsule_a_A.radius * result.contactNormal + p;
         result.deepestPointsB[0] = -1.f * capsule_b_B.radius * result.contactNormal + r;
      }
      else
      {
         result.deepestPointsA[0] = capsule_a_A.radius * result.contactNormal + p;
         result.deepestPointsA[1] = capsule_a_A.radius * result.contactNormal + q;
         result.deepestPointsB[0] = -1.f * capsule_b_B.radius * result.contactNormal + r;
         result.deepestPointsB[1] = -1.f * capsule_b_B.radius * result.contactNormal + s;
      }

      return result;
   }

   bool aabbAabb(
      const geometry::types::aabb_t & aabb_a,
      const geometry::types::aabb_t & aabb_b
   )
   {
      bool a_ahead_b = aabb_a.vertMin[0] > aabb_b.vertMax[0];
      bool a_behind_b = aabb_a.vertMax[0] < aabb_b.vertMin[0];

      bool a_left_b = aabb_a.vertMin[1] > aabb_b.vertMax[1];
      bool a_right_b = aabb_a.vertMax[1] < aabb_b.vertMin[1];

      bool a_above_b = aabb_a.vertMin[2] > aabb_b.vertMax[2];
      bool a_below_b = aabb_a.vertMax[2] < aabb_b.vertMin[2];

      return !(
         a_ahead_b || a_behind_b || a_left_b || a_right_b || a_above_b || a_below_b
      );
   }
}

}
