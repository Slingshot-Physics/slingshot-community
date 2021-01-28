#ifndef GEOMETRY_TYPE_CONVERTERS_HEADER
#define GEOMETRY_TYPE_CONVERTERS_HEADER

#include "data_model.h"
#include "geometry_types.hpp"

namespace geometry
{

namespace converters
{

   void from_pod(
      const data_vector3_t * data_in,
      Vector3 & data_out
   );

   void to_pod(
      const Vector3 & data_in,
      data_vector3_t * data_out
   );

   void from_pod(
      const data_vector4_t * data_in,
      Vector4 & data_out
   );

   void to_pod(
      const Vector4 & data_in,
      data_vector4_t * data_out
   );

   void from_pod(
      const data_matrix33_t * data_in,
      Matrix33 & data_out
   );

   void to_pod(
      const Matrix33 & data_in,
      data_matrix33_t * data_out
   );

   void from_pod(
      const data_isometricTransform_t * data_in,
      geometry::types::isometricTransform_t & data_out
   );

   void to_pod(
      const geometry::types::isometricTransform_t & data_in,
      data_isometricTransform_t * data_out
   );

   void from_pod(
      const data_transform_t * data_in,
      geometry::types::transform_t & data_out
   );

   void to_pod(
      const geometry::types::transform_t & data_in,
      data_transform_t * data_out
   );

   void from_pod(
      const data_meshTriangle_t * data_in,
      geometry::types::meshTriangle_t & data_out
   );

   void to_pod(
      const geometry::types::meshTriangle_t & data_in,
      data_meshTriangle_t * data_out
   );

   void from_pod(
      const data_triangle_t * data_in, geometry::types::triangle_t & data_out
   );

   void to_pod(
      const geometry::types::triangle_t & data_in, data_triangle_t * data_out
   );

   void from_pod(
      const data_tetrahedron_t * data_in,
      geometry::types::tetrahedron_t & data_out
   );

   void to_pod(
      const geometry::types::tetrahedron_t & data_in,
      data_tetrahedron_t * data_out
   );

   void from_pod(
      const data_shapeType_t * data_in,
      geometry::types::enumShape_t & data_out
   );

   void to_pod(
      const geometry::types::enumShape_t & data_in,
      data_shapeType_t * data_out
   );

   void from_pod(
      const data_minkowskiDiffSimplex_t * data_in,
      geometry::types::minkowskiDiffSimplex_t & data_out
   );

   void to_pod(
      const geometry::types::minkowskiDiffSimplex_t & data_in,
      data_minkowskiDiffSimplex_t * data_out
   );

   void from_pod(
      const data_polygon50_t * data_in,
      geometry::types::polygon50_t & data_out
   );

   void to_pod(
      const geometry::types::polygon50_t & data_in,
      data_polygon50_t * data_out
   );

   void from_pod(
      const data_leanNeighborTriangle_t * data_in,
      geometry::types::leanNeighborTriangle_t & data_out
   );

   void to_pod(
      const geometry::types::leanNeighborTriangle_t & data_in,
      data_leanNeighborTriangle_t * data_out
   );

   void from_pod(
      const data_triangleMesh_t * data_in,
      geometry::types::triangleMesh_t & data_out
   );

   void to_pod(
      const geometry::types::triangleMesh_t & data_in,
      data_triangleMesh_t * data_out
   );

   void from_pod(
      const data_gaussMapFace_t * data_in,
      geometry::types::gaussMapFace_t & data_out
   );

   void to_pod(
      const geometry::types::gaussMapFace_t & data_in,
      data_gaussMapFace_t * data_out
   );

   void from_pod(
      const data_gaussMapMesh_t * data_in,
      geometry::types::gaussMapMesh_t & data_out
   );

   void to_pod(
      const geometry::types::gaussMapMesh_t & data_in,
      data_gaussMapMesh_t * data_out
   );

   void from_pod(
      const data_shapeCapsule_t * data_in,
      geometry::types::shapeCapsule_t & data_out
   );

   void to_pod(
      const geometry::types::shapeCapsule_t & data_in,
      data_shapeCapsule_t * data_out
   );

   void from_pod(
      const data_shapeCube_t * data_in,
      geometry::types::shapeCube_t & data_out
   );

   void to_pod(
      const geometry::types::shapeCube_t & data_in,
      data_shapeCube_t * data_out
   );

   void from_pod(
      const data_shapeCylinder_t * data_in,
      geometry::types::shapeCylinder_t & data_out
   );

   void to_pod(
      const geometry::types::shapeCylinder_t & data_in,
      data_shapeCylinder_t * data_out
   );

   void from_pod(
      const data_shapeSphere_t * data_in,
      geometry::types::shapeSphere_t & data_out
   );

   void to_pod(
      const geometry::types::shapeSphere_t & data_in,
      data_shapeSphere_t * data_out
   );

   void from_pod(
      const data_shape_t * data_in,
      geometry::types::shape_t & data_out
   );

   void to_pod(
      const geometry::types::shape_t & data_in,
      data_shape_t * data_out
   );

   void from_pod(
      const data_testGjkInput_t * data_in,
      geometry::types::testGjkInput_t & data_out
   );

   void to_pod(
      const geometry::types::testGjkInput_t & data_in,
      data_testGjkInput_t * data_out
   );

   void from_pod(
      const data_testTriangleInput_t * data_in,
      geometry::types::testTriangleInput_t & data_out
   );

   void to_pod(
      const geometry::types::testTriangleInput_t & data_in,
      data_testTriangleInput_t * data_out
   );

   void from_pod(
      const data_testTetrahedronInput_t * data_in,
      geometry::types::testTetrahedronInput_t & data_out
   );

   void to_pod(
      const geometry::types::testTetrahedronInput_t & data_in,
      data_testTetrahedronInput_t * data_out
   );
}

}

#endif
