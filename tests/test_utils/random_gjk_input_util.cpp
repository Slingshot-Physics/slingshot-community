#include "random_gjk_input_util.hpp"

#include "geometry_type_converters.hpp"
#include "mesh.hpp"
#include "transform_utils.hpp"

namespace test_utils
{
   void load_meshes_and_polyhedra(
      const geometry::types::testGjkInput_t & data,
      geometry::types::convexPolyhedron_t & polyhedronA,
      geometry::types::convexPolyhedron_t & polyhedronB,
      geometry::types::triangleMesh_t & meshA,
      geometry::types::triangleMesh_t & meshB
   )
   {
      meshA = geometry::mesh::loadShapeMesh(data.shapeA);
      test_utils::convert_triangleMesh_to_convexPolyhedron(
         meshA, polyhedronA
      );

      meshB = geometry::mesh::loadShapeMesh(data.shapeB);
      test_utils::convert_triangleMesh_to_convexPolyhedron(
         meshB, polyhedronB
      );

      geometry::mesh::applyTransformation(data.transformA, meshA);
      geometry::mesh::applyTransformation(data.transformB, meshB);
   }

   void generate_random_gjk_input(
      const geometry::types::enumShape_t shapeAType,
      const geometry::types::enumShape_t shapeBType,
      geometry::types::testGjkInput_t & data,
      geometry::types::convexPolyhedron_t & polyhedronA,
      geometry::types::convexPolyhedron_t & polyhedronB,
      geometry::types::triangleMesh_t & meshA,
      geometry::types::triangleMesh_t & meshB
   )
   {
      data.transformA = test_utils::generate_random_transform();
      data.transformB = test_utils::generate_random_transform();
      data.shapeA = geometry::mesh::defaultShape(shapeAType);
      data.shapeB = geometry::mesh::defaultShape(shapeBType);
      load_meshes_and_polyhedra(data, polyhedronA, polyhedronB, meshA, meshB);
   }

}
