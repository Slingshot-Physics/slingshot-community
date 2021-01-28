#include "aabb_hull.hpp"
#include "support_functions.hpp"
#include "transform.hpp"

#include <algorithm>

namespace geometry
{
   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::convexPolyhedron_t & conv_polyhedron_C
   )
   {
      geometry::types::aabb_t aabb_hull_W;
      Vector3 & vert_max_W = aabb_hull_W.vertMax;
      Vector3 & vert_min_W = aabb_hull_W.vertMin;

      vert_max_W.Initialize(-__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__);
      vert_min_W.Initialize(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

      const Matrix33 A(trans_C_to_W.rotate * trans_C_to_W.scale);

      Vector3 trans_vert;
      for (unsigned int i = 0; i < conv_polyhedron_C.numVerts; ++i)
      {
         trans_vert = A * conv_polyhedron_C.verts[i];
         for (int j = 0; j < 3; ++j)
         {
            vert_max_W[j] = std::max(trans_vert[j], vert_max_W[j]);
            vert_min_W[j] = std::min(-trans_vert[j], vert_min_W[j]);
         }
      }

      vert_max_W += trans_C_to_W.translate;
      vert_min_W += trans_C_to_W.translate;

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCube_t & cube
   )
   {
      geometry::types::aabb_t aabb_hull_W;
      Vector3 & vert_max_W = aabb_hull_W.vertMax;
      Vector3 & vert_min_W = aabb_hull_W.vertMin;

      vert_max_W.Initialize(-__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__);
      vert_min_W.Initialize(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

      const float default_scale = 2.f;

      const Matrix33 A(trans_C_to_W.rotate * trans_C_to_W.scale);

      Vector3 trans_vert;
      for (unsigned int i = 0; i < 8; ++i)
      {
         const Vector3 vert_C = {
            (((i & 1) == 0) ? 1.f : -1.f) * cube.length / default_scale,
            (((i & 2) == 0) ? 1.f : -1.f) * cube.width / default_scale,
            (((i & 4) == 0) ? 1.f : -1.f) * cube.height / default_scale
         };
         trans_vert = A * vert_C;
         for (int j = 0; j < 3; ++j)
         {
            vert_max_W[j] = std::max(trans_vert[j], vert_max_W[j]);
            vert_min_W[j] = std::min(-trans_vert[j], vert_min_W[j]);
         }
      }

      vert_max_W += trans_C_to_W.translate;
      vert_min_W += trans_C_to_W.translate;

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeSphere_t & sphere
   )
   {
      Matrix33 support_dir_mat(
         trans_C_to_W.scale.transpose() * trans_C_to_W.rotate.transpose()
      );

      const Vector3 dirs[3] = {
         {1.f, 0.f, 0.f},
         {0.f, 1.f, 0.f},
         {0.f, 0.f, 1.f},
      };

      const Vector3 pos_supports_C[3] = {
         supportMapping(support_dir_mat * dirs[0], sphere).vert,
         supportMapping(support_dir_mat * dirs[1], sphere).vert,
         supportMapping(support_dir_mat * dirs[2], sphere).vert
      };

      const Vector3 neg_supports_C[3] = {
         supportMapping(-1.f * support_dir_mat * dirs[0], sphere).vert,
         supportMapping(-1.f * support_dir_mat * dirs[1], sphere).vert,
         supportMapping(-1.f * support_dir_mat * dirs[2], sphere).vert
      };

      const Vector3 pos_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[2])
      };

      const Vector3 neg_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[2])
      };

      geometry::types::aabb_t aabb_hull_W;
      for (int i = 0; i < 3; ++i)
      {
         aabb_hull_W.vertMax[i] = pos_supports_W[i][i];
         aabb_hull_W.vertMin[i] = neg_supports_W[i][i];
      }

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCapsule_t & capsule
   )
   {
      Matrix33 support_dir_mat(
         trans_C_to_W.scale.transpose() * trans_C_to_W.rotate.transpose()
      );

      const Vector3 dirs[3] = {
         {1.f, 0.f, 0.f},
         {0.f, 1.f, 0.f},
         {0.f, 0.f, 1.f},
      };

      const Vector3 pos_supports_C[3] = {
         supportMapping(support_dir_mat * dirs[0], capsule).vert,
         supportMapping(support_dir_mat * dirs[1], capsule).vert,
         supportMapping(support_dir_mat * dirs[2], capsule).vert
      };

      const Vector3 neg_supports_C[3] = {
         supportMapping(-1.f * support_dir_mat * dirs[0], capsule).vert,
         supportMapping(-1.f * support_dir_mat * dirs[1], capsule).vert,
         supportMapping(-1.f * support_dir_mat * dirs[2], capsule).vert
      };

      const Vector3 pos_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[2])
      };

      const Vector3 neg_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[2])
      };

      geometry::types::aabb_t aabb_hull_W;
      for (int i = 0; i < 3; ++i)
      {
         aabb_hull_W.vertMax[i] = pos_supports_W[i][i];
         aabb_hull_W.vertMin[i] = neg_supports_W[i][i];
      }

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shapeCylinder_t & cylinder
   )
   {
      Matrix33 support_dir_mat(
         trans_C_to_W.scale.transpose() * trans_C_to_W.rotate.transpose()
      );

      const Vector3 dirs[3] = {
         {1.f, 0.f, 0.f},
         {0.f, 1.f, 0.f},
         {0.f, 0.f, 1.f},
      };

      const Vector3 pos_supports_C[3] = {
         supportMapping(support_dir_mat * dirs[0], cylinder).vert,
         supportMapping(support_dir_mat * dirs[1], cylinder).vert,
         supportMapping(support_dir_mat * dirs[2], cylinder).vert
      };

      const Vector3 neg_supports_C[3] = {
         supportMapping(-1.f * support_dir_mat * dirs[0], cylinder).vert,
         supportMapping(-1.f * support_dir_mat * dirs[1], cylinder).vert,
         supportMapping(-1.f * support_dir_mat * dirs[2], cylinder).vert
      };

      const Vector3 pos_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[2])
      };

      const Vector3 neg_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[2])
      };

      geometry::types::aabb_t aabb_hull_W;
      for (int i = 0; i < 3; ++i)
      {
         aabb_hull_W.vertMax[i] = pos_supports_W[i][i];
         aabb_hull_W.vertMin[i] = neg_supports_W[i][i];
      }

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::transform_t & trans_C_to_W,
      const geometry::types::shape_t & shape
   )
   {
      geometry::types::aabb_t aabb_hull_W;
      switch(shape.shapeType)
      {
         case geometry::types::enumShape_t::CUBE:
         {
            aabb_hull_W = aabbHull(trans_C_to_W, shape.cube);
            break;
         }
         case geometry::types::enumShape_t::SPHERE:
         {
            aabb_hull_W = aabbHull(trans_C_to_W, shape.sphere);
            break;
         }
         case geometry::types::enumShape_t::CAPSULE:
         {
            aabb_hull_W = aabbHull(trans_C_to_W, shape.capsule);
            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            aabb_hull_W = aabbHull(trans_C_to_W, shape.cylinder);
            break;
         }
         default:
         {
            std::cout << "Couldn't generate AABB for shape type " << static_cast<int>(shape.shapeType) << "\n";
            break;
         }
      }

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::convexPolyhedron_t & conv_polyhedron_C
   )
   {
      geometry::types::aabb_t aabb_hull_W;
      Vector3 & vert_max_W = aabb_hull_W.vertMax;
      Vector3 & vert_min_W = aabb_hull_W.vertMin;

      vert_max_W.Initialize(-__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__);
      vert_min_W.Initialize(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

      Vector3 trans_vert;
      for (unsigned int i = 0; i < conv_polyhedron_C.numVerts; ++i)
      {
         trans_vert = trans_C_to_W.rotate * conv_polyhedron_C.verts[i];
         for (int j = 0; j < 3; ++j)
         {
            vert_max_W[j] = std::max(trans_vert[j], vert_max_W[j]);
            vert_min_W[j] = std::min(-trans_vert[j], vert_min_W[j]);
         }
      }

      vert_max_W += trans_C_to_W.translate;
      vert_min_W += trans_C_to_W.translate;

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCube_t & cube
   )
   {
      geometry::types::aabb_t aabb_hull_W;
      Vector3 & vert_max_W = aabb_hull_W.vertMax;
      Vector3 & vert_min_W = aabb_hull_W.vertMin;

      vert_max_W.Initialize(-__FLT_MAX__, -__FLT_MAX__, -__FLT_MAX__);
      vert_min_W.Initialize(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

      const float default_scale = 2.f;

      Vector3 trans_vert;
      for (unsigned int i = 0; i < 8; ++i)
      {
         const Vector3 vert_C = {
            (((i & 1) == 0) ? 1.f : -1.f) * cube.length / default_scale,
            (((i & 2) == 0) ? 1.f : -1.f) * cube.width / default_scale,
            (((i & 4) == 0) ? 1.f : -1.f) * cube.height / default_scale
         };
         trans_vert = trans_C_to_W.rotate * vert_C;
         for (int j = 0; j < 3; ++j)
         {
            vert_max_W[j] = std::max(trans_vert[j], vert_max_W[j]);
            vert_min_W[j] = std::min(-trans_vert[j], vert_min_W[j]);
         }
      }

      vert_max_W += trans_C_to_W.translate;
      vert_min_W += trans_C_to_W.translate;

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeSphere_t & sphere
   )
   {
      Matrix33 support_dir_mat(trans_C_to_W.rotate.transpose());

      const Vector3 dirs[3] = {
         {1.f, 0.f, 0.f},
         {0.f, 1.f, 0.f},
         {0.f, 0.f, 1.f},
      };

      const Vector3 pos_supports_C[3] = {
         supportMapping(support_dir_mat * dirs[0], sphere).vert,
         supportMapping(support_dir_mat * dirs[1], sphere).vert,
         supportMapping(support_dir_mat * dirs[2], sphere).vert
      };

      const Vector3 neg_supports_C[3] = {
         supportMapping(-1.f * support_dir_mat * dirs[0], sphere).vert,
         supportMapping(-1.f * support_dir_mat * dirs[1], sphere).vert,
         supportMapping(-1.f * support_dir_mat * dirs[2], sphere).vert
      };

      const Vector3 pos_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[2])
      };

      const Vector3 neg_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[2])
      };

      geometry::types::aabb_t aabb_hull_W;
      for (int i = 0; i < 3; ++i)
      {
         aabb_hull_W.vertMax[i] = pos_supports_W[i][i];
         aabb_hull_W.vertMin[i] = neg_supports_W[i][i];
      }

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCapsule_t & capsule
   )
   {
      Matrix33 support_dir_mat(trans_C_to_W.rotate.transpose());

      const Vector3 dirs[3] = {
         {1.f, 0.f, 0.f},
         {0.f, 1.f, 0.f},
         {0.f, 0.f, 1.f},
      };

      const Vector3 pos_supports_C[3] = {
         supportMapping(support_dir_mat * dirs[0], capsule).vert,
         supportMapping(support_dir_mat * dirs[1], capsule).vert,
         supportMapping(support_dir_mat * dirs[2], capsule).vert
      };

      const Vector3 neg_supports_C[3] = {
         supportMapping(-1.f * support_dir_mat * dirs[0], capsule).vert,
         supportMapping(-1.f * support_dir_mat * dirs[1], capsule).vert,
         supportMapping(-1.f * support_dir_mat * dirs[2], capsule).vert
      };

      const Vector3 pos_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[2])
      };

      const Vector3 neg_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[2])
      };

      geometry::types::aabb_t aabb_hull_W;
      for (int i = 0; i < 3; ++i)
      {
         aabb_hull_W.vertMax[i] = pos_supports_W[i][i];
         aabb_hull_W.vertMin[i] = neg_supports_W[i][i];
      }

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shapeCylinder_t & cylinder
   )
   {
      Matrix33 support_dir_mat(trans_C_to_W.rotate.transpose());

      const Vector3 dirs[3] = {
         {1.f, 0.f, 0.f},
         {0.f, 1.f, 0.f},
         {0.f, 0.f, 1.f},
      };

      const Vector3 pos_supports_C[3] = {
         supportMapping(support_dir_mat * dirs[0], cylinder).vert,
         supportMapping(support_dir_mat * dirs[1], cylinder).vert,
         supportMapping(support_dir_mat * dirs[2], cylinder).vert
      };

      const Vector3 neg_supports_C[3] = {
         supportMapping(-1.f * support_dir_mat * dirs[0], cylinder).vert,
         supportMapping(-1.f * support_dir_mat * dirs[1], cylinder).vert,
         supportMapping(-1.f * support_dir_mat * dirs[2], cylinder).vert
      };

      const Vector3 pos_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, pos_supports_C[2])
      };

      const Vector3 neg_supports_W[3] = {
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[0]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[1]),
         geometry::transform::forwardBound(trans_C_to_W, neg_supports_C[2])
      };

      geometry::types::aabb_t aabb_hull_W;
      for (int i = 0; i < 3; ++i)
      {
         aabb_hull_W.vertMax[i] = pos_supports_W[i][i];
         aabb_hull_W.vertMin[i] = neg_supports_W[i][i];
      }

      return aabb_hull_W;
   }

   geometry::types::aabb_t aabbHull(
      const geometry::types::isometricTransform_t & trans_C_to_W,
      const geometry::types::shape_t & shape
   )
   {
      geometry::types::aabb_t aabb_hull_W;
      switch(shape.shapeType)
      {
         case geometry::types::enumShape_t::CUBE:
         {
            aabb_hull_W = aabbHull(trans_C_to_W, shape.cube);
            break;
         }
         case geometry::types::enumShape_t::SPHERE:
         {
            aabb_hull_W = aabbHull(trans_C_to_W, shape.sphere);
            break;
         }
         case geometry::types::enumShape_t::CAPSULE:
         {
            aabb_hull_W = aabbHull(trans_C_to_W, shape.capsule);
            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            aabb_hull_W = aabbHull(trans_C_to_W, shape.cylinder);
            break;
         }
         default:
         {
            std::cout << "Couldn't generate AABB for shape type " << static_cast<int>(shape.shapeType) << "\n";
            break;
         }
      }

      return aabb_hull_W;
   }
}
