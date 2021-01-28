#ifndef EPA_HEADER
#define EPA_HEADER

#include "epa_types.hpp"
#include "geometry_types.hpp"
#include "matrix33.hpp"
#include "vector3.hpp"

#include "attitudeutils.hpp"
#include "geometry.hpp"
#include "gjk.hpp"
#include "mesh_ops.hpp"
#include "support_functions.hpp"
#include "transform.hpp"

#include <iostream>

namespace geometry
{

namespace epa
{
   struct EpaTerminationBase
   {
      // Determines if EPA should terminate based on the values of successive
      // EPA search directions.
      virtual bool operator()(
         const Vector3 & expand_dir, const Vector3 & prev_expand_dir
      ) const = 0;
   };

   // The default termination criteria. This empirically works well for
   // polyhedron-polyhedron collision geometry calculation.
   struct EpaTerminationDefault : EpaTerminationBase
   {
      EpaTerminationDefault(float sq_dist_threshold=1e-8f)
         : sq_dist_threshold_(sq_dist_threshold)
      { }

      virtual bool operator()(
         const Vector3 & expand_dir, const Vector3 & prev_expand_dir
      ) const
      {
         float sq_dist = (expand_dir - prev_expand_dir).magnitudeSquared();
         return sq_dist < sq_dist_threshold_;
      }

      private:
         // Used to compare the distances of successive EPA expansion
         // directions. Should be (0, 1e-3f)
         float sq_dist_threshold_;

   };

   // The termination criteria for smooth-smooth collision geometry.
   struct EpaTerminationSmooth: EpaTerminationBase
   {
      EpaTerminationSmooth(
         float sq_dist_threshold=1e-8f, float dot_threshold=(1.f - 1e-5f)
      )
         : sq_dist_threshold_(sq_dist_threshold)
         , dot_threshold_(dot_threshold)
      { }

      virtual bool operator()(
         const Vector3 & expand_dir, const Vector3 & prev_expand_dir
      ) const
      {
         float sq_dist = (expand_dir - prev_expand_dir).magnitudeSquared();
         float dot = expand_dir.unitVector().dot(prev_expand_dir.unitVector());
         return (sq_dist < sq_dist_threshold_) || (dot > dot_threshold_);
      }

      private:
         // Used to compare the distances of successive EPA expansion
         // directions. Should be (0, 1e-3f], empirically.
         float sq_dist_threshold_;

         // Used to compare the angle between the unit vectors of successive
         // EPA expansion directions. The default has been verified for smooth-
         // smooth shape collision geometry calculations.
         // Should be between (1.f - 1e-3f, 1.f], empirically.
         float dot_threshold_;

   };

   inline unsigned int leastSignificantAxis(Vector3 & v)
   {
      if (v[0] <= v[1] && v[0] <= v[2])
      {
         return 0;
      }
      if (v[1] <= v[0] && v[1] <= v[2])
      {
         return 1;
      }

      return 2;
   }

   // Expands GJK result using body A's frame of reference.
   // Artificially increases the size of the GJK simplex to generate a
   // tetrahedron.
   // I pilfered the logic from Allen Chou's blog:
   //    http://allenchou.net/2013/12/game-physics-contact-generation-epa/
   template <typename Transform_T, typename ShapeA_T, typename ShapeB_T>
   void expandGjkSimplex(
      const Transform_T & trans_A_to_W,
      const Transform_T & trans_B_to_W,
      const ShapeA_T & shape_a_A,
      const ShapeB_T & shape_b_B,
      geometry::types::minkowskiDiffSimplex_t & simplex
   )
   {
      Vector3 search_dirs_W[6];
      search_dirs_W[0][0] = 1.0f;
      search_dirs_W[1][0] = -1.0f;
      search_dirs_W[2][1] = 1.0f;
      search_dirs_W[3][1] = -1.0f;
      search_dirs_W[4][2] = 1.0f;
      search_dirs_W[5][2] = -1.0f;

      Vector3 basis[3];
      basis[0][0] = 1.0f;
      basis[1][1] = 1.0f;
      basis[2][2] = 1.0f;

      unsigned int smallest_axis;
      geometry::types::minkowskiDiffVertex_t support_md_W;
      Vector3 segment_W;
      Vector3 search_dir_W;
      Matrix33 R;

      MinkowskiDiffSupport<Transform_T, ShapeA_T, ShapeB_T> md_support(
         trans_A_to_W, trans_B_to_W, shape_a_A, shape_b_B
      );

      // Flow-through on these cases is intentional.
      switch(simplex.numVerts)
      {
         case 1:
            for (unsigned int i = 0; i < 6; ++i)
            {
               support_md_W = md_support(search_dirs_W[i]);

               if ((support_md_W.vert - simplex.verts[0]).magnitudeSquared() > 1e-7f)
               {
                  geometry::gjk::addVertexToSimplex(support_md_W, simplex);
                  break;
               }
            }
            // fall through
         case 2:
            segment_W = simplex.verts[1] - simplex.verts[0];
            smallest_axis = leastSignificantAxis(segment_W);
            search_dir_W = segment_W.crossProduct(basis[smallest_axis]);
            R = rodriguesRotation(segment_W, M_PI/3.0f);

            for (unsigned int i = 0; i < 6; ++i)
            {
               support_md_W = md_support(search_dirs_W[i]);

               if (
                  !geometry::segment::pointColinear(
                     simplex.verts[0], simplex.verts[1], support_md_W.vert
                  )
               )
               {
                  geometry::gjk::addVertexToSimplex(support_md_W, simplex);
                  break;
               }

               search_dir_W = R * search_dir_W;
            }
            // fall through
         case 3:
            search_dir_W = (
               simplex.verts[0] - simplex.verts[1]
            ).crossProduct(simplex.verts[2] - simplex.verts[1]);

            support_md_W = md_support(search_dir_W);

            if (
               !geometry::triangle::pointCoplanar(
                  simplex.verts[0],
                  simplex.verts[1],
                  simplex.verts[2],
                  support_md_W.vert
               )
            )
            {
               search_dir_W *= -1.0f;
               support_md_W = md_support(search_dir_W);
               geometry::gjk::addVertexToSimplex(support_md_W, simplex);
            }
            break;
         default:
            break;
      }
   }

   unsigned int findClosestTriangleId(
      const geometry::types::epa::epaMesh_t & mesh,
      const Vector3 & interior_point,
      geometry::types::pointBaryCoord_t & closest_point,
      Vector3 & closest_tri_normal
   );

   int deleteVisibleTriangles(
      geometry::types::epa::epaMesh_t & mesh,
      const Vector3 & interior_point,
      const Vector3 & epa_support_point
   );

   int addNewTriangles(
      geometry::types::epa::epaMesh_t & mesh, int new_vert_index
   );

   template <typename Transform_T, typename ShapeA_T, typename ShapeB_T>
   geometry::types::epaResult_t alg(
      const Transform_T & trans_A_to_W,
      const Transform_T & trans_B_to_W,
      const geometry::types::minkowskiDiffSimplex_t & tetra_md_simplex,
      const ShapeA_T & shape_a_A,
      const ShapeB_T & shape_b_B
   )
   {
      return alg(
         trans_A_to_W,
         trans_B_to_W,
         tetra_md_simplex,
         shape_a_A,
         shape_b_B,
         EpaTerminationDefault(1e-8f)
      );
   }

   template <typename Transform_T, typename ShapeA_T, typename ShapeB_T>
   geometry::types::epaResult_t smoothAlg(
      const Transform_T & trans_A_to_W,
      const Transform_T & trans_B_to_W,
      const geometry::types::minkowskiDiffSimplex_t & tetra_md_simplex,
      const ShapeA_T & shape_a_A,
      const ShapeB_T & shape_b_B,
      float dot_epsilon=(1.f - 1e-5f)
   )
   {
      return alg(
         trans_A_to_W,
         trans_B_to_W,
         tetra_md_simplex,
         shape_a_A,
         shape_b_B,
         EpaTerminationSmooth(1e-8f, dot_epsilon)
      );
   }

   // This implementation of EPA uses convex polyhedra that are in their
   // respective body frames. E.g. shape_a_A is a shape for body A in body
   // A's body frame, and shape_b_B is a shape for body B in body B's body
   // frame.
   // Notation:
   //    `_W` --> global or world coordinates (ENU)
   //    `_A` --> in body A's body frame
   //    `_B` --> in body B's body frame
   template <typename Transform_T, typename ShapeA_T, typename ShapeB_T, typename Termination_T>
   geometry::types::epaResult_t alg(
      const Transform_T & trans_A_to_W,
      const Transform_T & trans_B_to_W,
      const geometry::types::minkowskiDiffSimplex_t & tetra_md_simplex,
      const ShapeA_T & shape_a_A,
      const ShapeB_T & shape_b_B,
      Termination_T termination_criteria
   )
   {
      geometry::types::epaResult_t result;
      result.collided = true;

      MinkowskiDiffSupport<Transform_T, ShapeA_T, ShapeB_T> support_calculator(
         trans_A_to_W, trans_B_to_W, shape_a_A, shape_b_B
      );

      // Pre-calculate an interior point. This is guaranteed to be inside the
      // EPA mesh for the entire calculation.
      Vector3 interior_point_md_W = geometry::averageVertex(
         tetra_md_simplex.numVerts, tetra_md_simplex.verts
      );

      // Initialize the Minkowski Difference mesh with GJK's output.
      geometry::types::epa::epaMesh_t mesh_md_W(tetra_md_simplex);

      unsigned int closest_tri_id = 1001;
      geometry::types::pointBaryCoord_t closest_point_W;
      Vector3 closest_tri_normal_W;
      Vector3 prev_expand_dir_W(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

      unsigned int max_iters = 100;
      unsigned int iters = 0;
      for (iters = 0; iters < max_iters; ++iters)
      {
         closest_tri_id = findClosestTriangleId(
            mesh_md_W, interior_point_md_W, closest_point_W, closest_tri_normal_W
         );

         if (closest_tri_id >= mesh_md_W.triangles.size())
         {
            result.collided = false;
            break;
         }

         if (closest_point_W.point.hasNan() || closest_point_W.bary.hasNan())
         {
            result.collided = false;
            break;
         }

         Vector3 expand_dir_W = closest_point_W.point;

         if (expand_dir_W.magnitude() < 1e-6f)
         {
            expand_dir_W = closest_tri_normal_W;
         }

         // Terminate successfully with good collision geometry if the
         // termination criteria say we can.
         if (termination_criteria(expand_dir_W, prev_expand_dir_W))
         {
            break;
         }

         prev_expand_dir_W = expand_dir_W;

         gjk_mdVertex_t support_md_vert = support_calculator(expand_dir_W);

         if (mesh_md_W.vertExists(support_md_vert))
         {
            continue;
         }

         // Remove triangles that can see the support point.
         int num_deleted_triangles = deleteVisibleTriangles(
            mesh_md_W, interior_point_md_W, support_md_vert.vert
         );

         // If no triangles were deleted, don't bother adding the vertex or
         // trying to add new triangles to the mesh.
         if (num_deleted_triangles == 0)
         {
            continue;
         }

         // Add the support point to the set of vertices on the EPA mesh.
         int new_vert_index = mesh_md_W.addVert(support_md_vert);

         // Ran out of space in the mesh or tried to add a duplicate point,
         // respectively.
         if (new_vert_index == -2)
         {
            break;
         }

         int added_triangle = addNewTriangles(mesh_md_W, new_vert_index);

         // Add new triangles in place of the hole just created.
         if (added_triangle < 0)
         {
            result.collided = false;
            break;
         }

         if (mesh_md_W.triangles.size() == 0)
         {
            result.collided = false;
            break;
         }
      }

      if (result.collided)
      {
         const Vector4 & min_norm_bary = closest_point_W.bary;

         // Vertex IDs on the MD mesh that belong to the triangle closest to
         // the origin.
         unsigned int vertId0 = mesh_md_W.triangles[closest_tri_id].vertIds[0];
         unsigned int vertId1 = mesh_md_W.triangles[closest_tri_id].vertIds[1];
         unsigned int vertId2 = mesh_md_W.triangles[closest_tri_id].vertIds[2];

         result.bodyAContactPoint = (
            geometry::transform::forwardBound(trans_A_to_W, mesh_md_W.mdVerts[vertId0].bodyAVert) * min_norm_bary[0] +
            geometry::transform::forwardBound(trans_A_to_W, mesh_md_W.mdVerts[vertId1].bodyAVert) * min_norm_bary[1] +
            geometry::transform::forwardBound(trans_A_to_W, mesh_md_W.mdVerts[vertId2].bodyAVert) * min_norm_bary[2]
         );

         result.bodyBContactPoint = (
            geometry::transform::forwardBound(trans_B_to_W, mesh_md_W.mdVerts[vertId0].bodyBVert) * min_norm_bary[0] +
            geometry::transform::forwardBound(trans_B_to_W, mesh_md_W.mdVerts[vertId1].bodyBVert) * min_norm_bary[1] +
            geometry::transform::forwardBound(trans_B_to_W, mesh_md_W.mdVerts[vertId2].bodyBVert) * min_norm_bary[2]
         );

         result.p = prev_expand_dir_W;
      }

      return result;
   }

   template <typename Transform_T, typename ShapeA_T, typename ShapeB_T>
   geometry::types::epa::epaMesh_t alg_debug(
      const Transform_T & trans_A_to_W,
      const Transform_T & trans_B_to_W,
      const geometry::types::minkowskiDiffSimplex_t & tetra_md_simplex,
      const ShapeA_T & shape_a_A,
      const ShapeB_T & shape_b_B
   )
   {
      return alg_debug(
         trans_A_to_W,
         trans_B_to_W,
         tetra_md_simplex,
         shape_a_A,
         shape_b_B,
         EpaTerminationDefault(1e-8f)
      );
   }

   // This implementation of EPA uses convex polyhedra that are in their
   // respective body frames. E.g. shape_a_A is a shape for body A in body
   // A's body frame, and shape_b_B is a shape for body B in body B's body
   // frame.
   // Notation:
   //    `_W` --> global or world coordinates (ENU)
   //    `_A` --> in body A's body frame
   //    `_B` --> in body B's body frame
   template <typename Transform_T, typename ShapeA_T, typename ShapeB_T, typename Termination_T>
   geometry::types::epa::epaMesh_t alg_debug(
      const Transform_T & trans_A_to_W,
      const Transform_T & trans_B_to_W,
      const geometry::types::minkowskiDiffSimplex_t & tetra_md_simplex,
      const ShapeA_T & shape_a_A,
      const ShapeB_T & shape_b_B,
      Termination_T termination_criteria
   )
   {
      std::cout << "----------epa--------------\n";
      
      MinkowskiDiffSupport<Transform_T, ShapeA_T, ShapeB_T> support_calculator(
         trans_A_to_W, trans_B_to_W, shape_a_A, shape_b_B
      );

      // Pre-calculate an interior point. This is guaranteed to be inside the
      // EPA mesh for the entire calculation.
      Vector3 interior_point_md_W = geometry::averageVertex(
         tetra_md_simplex.numVerts, tetra_md_simplex.verts
      );

      // Initialize the Minkowski Difference mesh with GJK's output.
      geometry::types::epa::epaMesh_t mesh_md_W(tetra_md_simplex);

      unsigned int closest_tri_id = 1001;
      geometry::types::pointBaryCoord_t closest_point_W;
      Vector3 closest_tri_normal_W;
      Vector3 prev_expand_dir_W(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

      unsigned int max_iters = 100;
      unsigned int iters = 0;
      for (iters = 0; iters < max_iters; ++iters)
      {
         closest_tri_id = findClosestTriangleId(
            mesh_md_W, interior_point_md_W, closest_point_W, closest_tri_normal_W
         );

         if (closest_tri_id >= mesh_md_W.triangles.size())
         {
            break;
         }

         if (closest_point_W.point.hasNan() || closest_point_W.bary.hasNan())
         {
            break;
         }

         Vector3 expand_dir_W = closest_point_W.point;

         if (expand_dir_W.magnitude() < 1e-6f)
         {
            std::cout << "expanding with triangle normal\n";
            expand_dir_W = closest_tri_normal_W;
         }

         // Terminate successfully with good collision geometry if the
         // termination criteria say we can.
         if (termination_criteria(expand_dir_W, prev_expand_dir_W))
         {
            break;
         }

         prev_expand_dir_W = expand_dir_W;

         gjk_mdVertex_t support_md_vert = support_calculator(expand_dir_W);

         if (mesh_md_W.vertExists(support_md_vert))
         {
            continue;
         }

         // Remove triangles that can see the support point.
         int num_deleted_triangles = deleteVisibleTriangles(
            mesh_md_W, interior_point_md_W, support_md_vert.vert
         );

         // If no triangles were deleted, don't bother adding the vertex or
         // trying to add new triangles to the mesh.
         if (num_deleted_triangles == 0)
         {
            continue;
         }

         // Add the support point to the set of vertices on the EPA mesh.
         int new_vert_index = mesh_md_W.addVert(support_md_vert);

         std::cout << "Added new EPA point " << support_md_vert.vert << "\n";

         // Ran out of space in the mesh or tried to add a duplicate point,
         // respectively.
         if (new_vert_index == -2)
         {
            break;
         }

         int added_triangle = addNewTriangles(mesh_md_W, new_vert_index);

         // Add new triangles in place of the hole just created.
         if (added_triangle < 0)
         {
            break;
         }

         if (mesh_md_W.triangles.size() == 0)
         {
            break;
         }
      }

      return mesh_md_W;
   }

}

}

#endif
