#include "shape_features.hpp"

#include "gauss_map.hpp"
#include "polygon.hpp"
#include "support_functions.hpp"
#include "transform.hpp"

#include <cmath>

namespace geometry
{
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::gaussMapMesh_t & gauss_mesh_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      // The dot product that would normally be
      //    dot(normal_mesh_W, query_normal_W)
      // is broken up into two pieces to take advantage of normals in collider
      // space.
      // The dot product becomes
      //    dot(
      //       (R_C_to_W * (~S_C)^T * normal_mesh_C) / ((~S_C) * normal_mesh_C).magnitude(),
      //       query_normal_W
      //    )
      // And by associativity of matrix multiplication/commutability of the dot
      // product, the matrix multiplications in the numerator of normal_mesh_W
      // can be applied to query_normal_W (to save matrix multiplies at
      // run-time).
      const Vector3 & query_normal_W = query_plane_W.normal;
      const Matrix33 P(~trans_M_to_W.scale);
      const Vector3 query = (
         P * (trans_M_to_W.rotate.transpose() * query_normal_W)
      );

      unsigned int best_face_id = 0;
      float max_dot = -1.f;

      for (unsigned int i = 0; i < gauss_mesh_M.numFaces; ++i)
      {
         const Vector3 & face_normal_M = gauss_mesh_M.faces[i].normal;
         const Vector3 & scaled_face_normal = (
            face_normal_M / (P * face_normal_M).magnitude()
         );

         float temp_dot = scaled_face_normal.dot(query);

         if (temp_dot > max_dot)
         {
            best_face_id = i;
            max_dot = temp_dot;
         }
      }

      geometry::types::polyhedronFeature_t face_W;
      face_W.shape = geometry::mesh::polygonFromFace(gauss_mesh_M, best_face_id);

      geometry::polygon::applyTransformation(trans_M_to_W, face_W.shape);

      const Matrix33 A_inv_trans(trans_M_to_W.rotate * (P.transpose()));
      face_W.normal = A_inv_trans * gauss_mesh_M.faces[best_face_id].normal;
      face_W.normal.Normalize();

      return face_W;
   }

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::shapeCube_t & cube_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      // The dot product that would normally be
      //    dot(normal_mesh_W, query_normal_W)
      // is broken up into two pieces to take advantage of normals in collider
      // space.
      // The dot product becomes
      //    dot(
      //       (R_C_to_W * (~S_C)^T * normal_mesh_C) / ((~S_C) * normal_mesh_C).magnitude(),
      //       query_normal_W
      //    )
      // And by associativity of matrix multiplication/commutability of the dot
      // product, the matrix multiplications in the numerator of normal_mesh_W
      // can be applied to query_normal_W (to save matrix multiplies at
      // run-time).
      const Vector3 & query_normal_W = query_plane_W.normal;
      const Matrix33 P(~trans_M_to_W.scale);
      const Vector3 query = (
         P * (trans_M_to_W.rotate.transpose() * query_normal_W)
      );

      unsigned int best_face_id = 0;
      float max_dot = -1.f;

      // 0: +x_hat ( 1,  1,  1), ( 1,  1, -1), ( 1, -1, -1), ( 1, -1,  1)
      // 1: -x_hat (-1,  1,  1), (-1,  1, -1), (-1, -1, -1), (-1, -1,  1)
      // 2: +y_hat ( 1,  1,  1), ( 1,  1, -1), (-1,  1, -1), (-1,  1,  1)
      // 3: -y_hat ( 1, -1,  1), ( 1, -1, -1), (-1, -1, -1), (-1, -1,  1)
      // 4: +z_hat ( 1,  1,  1), ( 1, -1,  1), (-1, -1,  1), (-1,  1,  1)
      // 5: -z_hat ( 1,  1, -1), ( 1, -1, -1), (-1, -1, -1), (-1,  1, -1)

      Vector3 best_face_normal_M;
      Vector3 face_normal_M;
      for (unsigned int i = 0; i < 6; ++i)
      {
         face_normal_M.Initialize(0.f, 0.f, 0.f);
         face_normal_M[i / 2] = (((i & 1) == 0) ? 1.f : -1.f);
         float temp_dot = face_normal_M.dot(query) / (P * face_normal_M).magnitude();
         if (temp_dot > max_dot)
         {
            max_dot = temp_dot;
            best_face_id = i;
            best_face_normal_M = face_normal_M;
         }
      }

      Vector3 dimensions(cube_M.length, cube_M.width, cube_M.height);

      geometry::types::polyhedronFeature_t face_W;
      face_W.shape.numVerts = 4;
      float dir_sign = ((best_face_id & 1) == 0) ? 1.f : -1.f;
      for (unsigned int i = 0; i < 4; ++i)
      {
         unsigned int ind1 = ((best_face_id / 2) + 1) % 3;
         unsigned int ind2 = ((best_face_id / 2) + 2) % 3;
         face_W.shape.verts[i][ind1] = (((i & 1) == 0) ? 1.f : -1.f) * dimensions[ind1] / 2.f;
         face_W.shape.verts[i][ind2] = (((i & 2) == 0) ? 1.f : -1.f) * dimensions[ind2] / 2.f;
         face_W.shape.verts[i][best_face_id / 2] = dir_sign * dimensions[best_face_id / 2] / 2.f;
      }

      geometry::polygon::applyTransformation(trans_M_to_W, face_W.shape);

      const Matrix33 A_inv_trans(trans_M_to_W.rotate * (P.transpose()));
      face_W.normal = A_inv_trans * best_face_normal_M;
      face_W.normal.Normalize();

      return face_W;
   }

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::shapeCapsule_t & capsule_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      Vector3 query_normal_M(
         trans_M_to_W.scale.transpose() * \
         trans_M_to_W.rotate.transpose() * \
         query_plane_W.normal
      );
      query_normal_M.Normalize();

      Vector3 a_M(0.f, 0.f, -capsule_M.height / 2.f);
      Vector3 b_M(0.f, 0.f,  capsule_M.height / 2.f);

      // u_M is the vector along the vertical axis of the cylinder.
      // s_M is the normal to the edge that's most aligned with the query
      // plane's normal.
      Vector3 u_M = b_M - a_M;
      Vector3 s_M = (
         u_M.crossProduct(query_normal_M)
      ).crossProduct(u_M);

      Matrix33 normal_trans_mat(
         trans_M_to_W.rotate * (~trans_M_to_W.scale).transpose()
      );

      Vector3 s_hat_M = s_M.unitVector();
      Vector3 s_hat_W(normal_trans_mat * s_hat_M);
      s_hat_W.Normalize();

      float dot = s_hat_W.dot(query_plane_W.normal);

      geometry::types::polyhedronFeature_t face_W;

      // Range where the edge normal is considered parallel enough to the
      // query plane's normal.
      if (fabs(dot) > 1.f - 1e-4f)
      {
         face_W.shape.numVerts = 2;
         face_W.shape.verts[0] = geometry::transform::forwardBound(
            trans_M_to_W, a_M + s_hat_M * capsule_M.radius
         );
         face_W.shape.verts[1] = geometry::transform::forwardBound(
            trans_M_to_W, b_M + s_hat_M * capsule_M.radius
         );
         face_W.normal = s_hat_W;
      }
      else
      {
         Vector3 support_vert_M = geometry::supportMapping(
            query_normal_M, capsule_M
         ).vert;

         face_W.shape.numVerts = 1;
         face_W.shape.verts[0] = geometry::transform::forwardBound(
            trans_M_to_W, support_vert_M
         );

         face_W.normal = query_plane_W.normal;
      }

      return face_W;
   }

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::shapeCylinder_t & cylinder_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      Vector3 query_normal_M(
         trans_M_to_W.scale.transpose() * \
         trans_M_to_W.rotate.transpose() * \
         query_plane_W.normal
      );
      query_normal_M.Normalize();

      Vector3 a_M(0.f, 0.f, -cylinder_M.height / 2.f);
      Vector3 b_M(0.f, 0.f,  cylinder_M.height / 2.f);

      // u_M is the vector along the vertical axis of the cylinder.
      // s_M is the normal to the edge that's most aligned with the query
      // plane's normal.
      Vector3 u_M = b_M - a_M;
      Vector3 s_M = (
         u_M.crossProduct(query_normal_M)
      ).crossProduct(u_M);

      Matrix33 normal_trans_mat(
         trans_M_to_W.rotate * (~trans_M_to_W.scale).transpose()
      );

      Vector3 s_hat_M = s_M.unitVector();
      Vector3 s_hat_W(normal_trans_mat * s_hat_M);
      s_hat_W.Normalize();

      Vector3 face_a_hat_W(normal_trans_mat * Vector3(0.f, 0.f, 1.f));
      face_a_hat_W.Normalize();

      // Face A's normal is in the positive z-hat direction in collider space.
      // Face B's normal is in the negative z-hat direction in collider space.
      float edge_dot = s_hat_W.dot(query_plane_W.normal);
      float face_a_dot = face_a_hat_W.dot(query_plane_W.normal);
      float face_b_dot = -1.f * face_a_dot;

      geometry::types::polyhedronFeature_t face_W;

      if (edge_dot > fabs(face_a_dot))
      {
         face_W.shape.numVerts = 2;
         face_W.shape.verts[0] = geometry::transform::forwardBound(
            trans_M_to_W, a_M + s_hat_M * cylinder_M.radius
         );
         face_W.shape.verts[1] = geometry::transform::forwardBound(
            trans_M_to_W, b_M + s_hat_M * cylinder_M.radius
         );
         face_W.normal = s_hat_W;
      }
      else
      {
         face_W.shape.numVerts = 6;

         for (int i = 0; i < 6; ++i)
         {
            Vector3 face_vert_M(
               cylinder_M.radius * cosf(2 * i * M_PI / 6.f),
               cylinder_M.radius * sinf(2 * i * M_PI / 6.f),
               (cylinder_M.height / 2.f) * ((face_a_dot > face_b_dot) ? 1.f : -1.f)
            );
            face_W.shape.verts[i] = geometry::transform::forwardBound(
               trans_M_to_W, face_vert_M
            );
         }

         face_W.normal = face_a_hat_W;
         face_W.normal *= ((face_a_dot > face_b_dot) ? 1.f : -1.f);
      }

      return face_W;
   }

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::shapeSphere_t & sphere_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      (void)trans_M_to_W;
      (void)sphere_M;
      geometry::types::polyhedronFeature_t face_W;
      face_W.shape.numVerts = 1;
      face_W.shape.verts[0] = query_plane_W.point;

      face_W.normal = query_plane_W.normal;

      return face_W;
   }

/// isometric transform stuff

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::gaussMapMesh_t & gauss_mesh_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      const Vector3 & query_normal_W = query_plane_W.normal;
      const Vector3 query = (
         trans_M_to_W.rotate.transpose() * query_normal_W
      );

      unsigned int best_face_id = 0;
      float max_dot = -1.f;

      for (unsigned int i = 0; i < gauss_mesh_M.numFaces; ++i)
      {
         const Vector3 & face_normal_M = gauss_mesh_M.faces[i].normal;
         const Vector3 & scaled_face_normal = face_normal_M.unitVector();

         float temp_dot = scaled_face_normal.dot(query);

         if (temp_dot > max_dot)
         {
            best_face_id = i;
            max_dot = temp_dot;
         }
      }

      geometry::types::polyhedronFeature_t face_W;
      face_W.shape = geometry::mesh::polygonFromFace(gauss_mesh_M, best_face_id);

      geometry::polygon::applyTransformation(trans_M_to_W, face_W.shape);

      face_W.normal = trans_M_to_W.rotate * gauss_mesh_M.faces[best_face_id].normal;
      face_W.normal.Normalize();

      return face_W;
   }

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::shapeCube_t & cube_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      const Vector3 & query_normal_W = query_plane_W.normal;
      const Vector3 query = (
         trans_M_to_W.rotate.transpose() * query_normal_W
      );

      unsigned int best_face_id = 0;
      float max_dot = -1.f;

      // 0: +x_hat ( 1,  1,  1), ( 1,  1, -1), ( 1, -1, -1), ( 1, -1,  1)
      // 1: -x_hat (-1,  1,  1), (-1,  1, -1), (-1, -1, -1), (-1, -1,  1)
      // 2: +y_hat ( 1,  1,  1), ( 1,  1, -1), (-1,  1, -1), (-1,  1,  1)
      // 3: -y_hat ( 1, -1,  1), ( 1, -1, -1), (-1, -1, -1), (-1, -1,  1)
      // 4: +z_hat ( 1,  1,  1), ( 1, -1,  1), (-1, -1,  1), (-1,  1,  1)
      // 5: -z_hat ( 1,  1, -1), ( 1, -1, -1), (-1, -1, -1), (-1,  1, -1)

      Vector3 best_face_normal_M;
      Vector3 face_normal_M;
      for (unsigned int i = 0; i < 6; ++i)
      {
         face_normal_M.Initialize(0.f, 0.f, 0.f);
         face_normal_M[i / 2] = (((i & 1) == 0) ? 1.f : -1.f);
         float temp_dot = face_normal_M.dot(query) / face_normal_M.magnitude();
         if (temp_dot > max_dot)
         {
            max_dot = temp_dot;
            best_face_id = i;
            best_face_normal_M = face_normal_M;
         }
      }

      Vector3 dimensions(cube_M.length, cube_M.width, cube_M.height);

      geometry::types::polyhedronFeature_t face_W;
      face_W.shape.numVerts = 4;
      float dir_sign = ((best_face_id & 1) == 0) ? 1.f : -1.f;
      for (unsigned int i = 0; i < 4; ++i)
      {
         unsigned int ind1 = ((best_face_id / 2) + 1) % 3;
         unsigned int ind2 = ((best_face_id / 2) + 2) % 3;
         face_W.shape.verts[i][ind1] = (((i & 1) == 0) ? 1.f : -1.f) * dimensions[ind1] / 2.f;
         face_W.shape.verts[i][ind2] = (((i & 2) == 0) ? 1.f : -1.f) * dimensions[ind2] / 2.f;
         face_W.shape.verts[i][best_face_id / 2] = dir_sign * dimensions[best_face_id / 2] / 2.f;
      }

      geometry::polygon::applyTransformation(trans_M_to_W, face_W.shape);

      face_W.normal = trans_M_to_W.rotate * best_face_normal_M;
      face_W.normal.Normalize();

      return face_W;
   }

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::shapeCapsule_t & capsule_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      Vector3 query_normal_M(
         trans_M_to_W.rotate.transpose() * query_plane_W.normal
      );
      query_normal_M.Normalize();

      Vector3 a_M(0.f, 0.f, -capsule_M.height / 2.f);
      Vector3 b_M(0.f, 0.f,  capsule_M.height / 2.f);

      // u_M is the vector along the vertical axis of the cylinder.
      // s_M is the normal to the edge that's most aligned with the query
      // plane's normal.
      Vector3 u_M = b_M - a_M;
      Vector3 s_M = (
         u_M.crossProduct(query_normal_M)
      ).crossProduct(u_M);

      Matrix33 normal_trans_mat(trans_M_to_W.rotate);

      Vector3 s_hat_M = s_M.unitVector();
      Vector3 s_hat_W(normal_trans_mat * s_hat_M);
      s_hat_W.Normalize();

      float dot = s_hat_W.dot(query_plane_W.normal);

      geometry::types::polyhedronFeature_t face_W;

      // Range where the edge normal is considered parallel enough to the
      // query plane's normal.
      if (fabs(dot) > 1.f - 1e-4f)
      {
         face_W.shape.numVerts = 2;
         face_W.shape.verts[0] = geometry::transform::forwardBound(
            trans_M_to_W, a_M + s_hat_M * capsule_M.radius
         );
         face_W.shape.verts[1] = geometry::transform::forwardBound(
            trans_M_to_W, b_M + s_hat_M * capsule_M.radius
         );
         face_W.normal = s_hat_W;
      }
      else
      {
         Vector3 support_vert_M = geometry::supportMapping(
            query_normal_M, capsule_M
         ).vert;

         face_W.shape.numVerts = 1;
         face_W.shape.verts[0] = geometry::transform::forwardBound(
            trans_M_to_W, support_vert_M
         );

         face_W.normal = query_plane_W.normal;
      }

      return face_W;
   }

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::shapeCylinder_t & cylinder_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      Vector3 query_normal_M(
         trans_M_to_W.rotate.transpose() * query_plane_W.normal
      );
      query_normal_M.Normalize();

      Vector3 a_M(0.f, 0.f, -cylinder_M.height / 2.f);
      Vector3 b_M(0.f, 0.f,  cylinder_M.height / 2.f);

      // u_M is the vector along the vertical axis of the cylinder.
      // s_M is the normal to the edge that's most aligned with the query
      // plane's normal.
      Vector3 u_M = b_M - a_M;
      Vector3 s_M = (
         u_M.crossProduct(query_normal_M)
      ).crossProduct(u_M);

      Matrix33 normal_trans_mat(trans_M_to_W.rotate);

      Vector3 s_hat_M = s_M.unitVector();
      Vector3 s_hat_W(normal_trans_mat * s_hat_M);
      s_hat_W.Normalize();

      Vector3 face_a_hat_W(normal_trans_mat * Vector3(0.f, 0.f, 1.f));
      face_a_hat_W.Normalize();

      // Face A's normal is in the positive z-hat direction in collider space.
      // Face B's normal is in the negative z-hat direction in collider space.
      float edge_dot = s_hat_W.dot(query_plane_W.normal);
      float face_a_dot = face_a_hat_W.dot(query_plane_W.normal);
      float face_b_dot = -1.f * face_a_dot;

      geometry::types::polyhedronFeature_t face_W;

      if (edge_dot > fabs(face_a_dot))
      {
         face_W.shape.numVerts = 2;
         face_W.shape.verts[0] = geometry::transform::forwardBound(
            trans_M_to_W, a_M + s_hat_M * cylinder_M.radius
         );
         face_W.shape.verts[1] = geometry::transform::forwardBound(
            trans_M_to_W, b_M + s_hat_M * cylinder_M.radius
         );
         face_W.normal = s_hat_W;
      }
      else
      {
         face_W.shape.numVerts = 6;

         for (int i = 0; i < 6; ++i)
         {
            Vector3 face_vert_M(
               cylinder_M.radius * cosf(2 * i * M_PI / 6.f),
               cylinder_M.radius * sinf(2 * i * M_PI / 6.f),
               (cylinder_M.height / 2.f) * ((face_a_dot > face_b_dot) ? 1.f : -1.f)
            );
            face_W.shape.verts[i] = geometry::transform::forwardBound(
               trans_M_to_W, face_vert_M
            );
         }

         face_W.normal = face_a_hat_W;
         face_W.normal *= ((face_a_dot > face_b_dot) ? 1.f : -1.f);
      }

      return face_W;
   }

   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::shapeSphere_t & sphere_M,
      const geometry::types::plane_t & query_plane_W
   )
   {
      (void)trans_M_to_W;
      (void)sphere_M;
      geometry::types::polyhedronFeature_t face_W;
      face_W.shape.numVerts = 1;
      face_W.shape.verts[0] = query_plane_W.point;

      face_W.normal = query_plane_W.normal;

      return face_W;
   }
}
