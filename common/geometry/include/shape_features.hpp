#ifndef SHAPE_FEATURES_HEADER
#define SHAPE_FEATURES_HEADER

#include "geometry_types.hpp"

namespace geometry
{
   // Finds the feature on a gauss map mesh that's most parallel with the query
   // plane. The query plane should contain a normal and a point on the surface
   // of the gauss map mesh. The features of a gauss map that are most parallel
   // to a plane are faces.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::gaussMapMesh_t & gauss_mesh_M,
      const geometry::types::plane_t & query_plane_W
   );

   // Finds the feature on a cube that's most parallel with the query plane.
   // The query plane should contain a normal and a point on the surface of the
   // cube.
   // The two possible features for a cube are an edge or a point on either
   // end of the cube.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::shapeCube_t & cube_M,
      const geometry::types::plane_t & query_plane_W
   );

   // Finds the feature on a capsule that's most parallel with the query plane.
   // The query plane should contain a normal and a point on the surface of the
   // capsule.
   // The two possible features for a capsule are an edge or a point on either
   // end of the capsule.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::shapeCapsule_t & capsule_M,
      const geometry::types::plane_t & query_plane_W
   );

   // Finds the feature on a cylinder that's most parallel with the query
   // plane. The query plane should contain a normal and a point on the surface
   // of the cylinder.
   // There are only two possible features for a cylinder: an edge (parallel to
   // the axis of vertical symmetry), or a face.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::shapeCylinder_t & cylinder_M,
      const geometry::types::plane_t & query_plane_W
   );

   // Extraordinarily basic and mostly provided for function overloading with
   // function templates. It just returns the point on the query plane.
   // The query plane should contain a normal and a point on the surface of the
   // sphere.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::transform_t & trans_M_to_W,
      const geometry::types::shapeSphere_t & sphere_M,
      const geometry::types::plane_t & query_plane_W
   );

/// isometric transform stuff

   // Finds the feature on a gauss map mesh that's most parallel with the query
   // plane. The query plane should contain a normal and a point on the surface
   // of the gauss map mesh. The features of a gauss map that are most parallel
   // to a plane are faces.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::gaussMapMesh_t & gauss_mesh_M,
      const geometry::types::plane_t & query_plane_W
   );

   // Finds the feature on a cube that's most parallel with the query plane.
   // The query plane should contain a normal and a point on the surface of the
   // cube.
   // The two possible features for a cube are an edge or a point on either
   // end of the cube.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::shapeCube_t & cube_M,
      const geometry::types::plane_t & query_plane_W
   );

   // Finds the feature on a capsule that's most parallel with the query plane.
   // The query plane should contain a normal and a point on the surface of the
   // capsule.
   // The two possible features for a capsule are an edge or a point on either
   // end of the capsule.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::shapeCapsule_t & capsule_M,
      const geometry::types::plane_t & query_plane_W
   );

   // Finds the feature on a cylinder that's most parallel with the query
   // plane. The query plane should contain a normal and a point on the surface
   // of the cylinder.
   // There are only two possible features for a cylinder: an edge (parallel to
   // the axis of vertical symmetry), or a face.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::shapeCylinder_t & cylinder_M,
      const geometry::types::plane_t & query_plane_W
   );

   // Extraordinarily basic and mostly provided for function overloading with
   // function templates. It just returns the point on the query plane.
   // The query plane should contain a normal and a point on the surface of the
   // sphere.
   geometry::types::polyhedronFeature_t mostParallelFeature(
      const geometry::types::isometricTransform_t & trans_M_to_W,
      const geometry::types::shapeSphere_t & sphere_M,
      const geometry::types::plane_t & query_plane_W
   );
}

#endif
