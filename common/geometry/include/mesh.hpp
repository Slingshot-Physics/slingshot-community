#ifndef MESH_HEADER
#define MESH_HEADER

#include "data_enums.h"
#include "data_triangle_mesh.h"
#include "geometry_types.hpp"

namespace geometry
{

namespace mesh
{
   struct pointOnMesh_t
   {
      Vector3 point;
      Vector4 bary;
      unsigned int triangleId;
   };

   void scaleMesh(float scale, geometry::types::triangleMesh_t & mesh_in);

   // Scales the vertices about the geometric center of the mesh.
   void scaleMesh(
      const Vector3 & scale, geometry::types::triangleMesh_t & mesh_in
   );

   // Scales the vertices about the geometric center of the mesh.
   void scaleMesh(
      const Matrix33 & scale, geometry::types::triangleMesh_t & mesh_in
   );

   // Rotates the vertices about the geometric center of the mesh.
   // This could be overloaded to take in a point to rotate about.
   void rotateMesh(
      const Vector3 & rpy, geometry::types::triangleMesh_t & mesh_in
   );

   // Rotates the vertices about the geometric center of the mesh.
   // This could be overloaded to take in a point to rotate about.
   void rotateMesh(
      const Matrix33 & R_poly_to_world, geometry::types::triangleMesh_t & mesh_in
   );

   // Translates all of the vertices based on the offset vector.
   void translateMesh(
      const Vector3 & offset, geometry::types::triangleMesh_t & mesh_in
   );

   // Applies a transformation to a mesh. The result of the transformation is
   // saved back into the original mesh.
   void applyTransformation(
      const geometry::types::transform_t & transform,
      geometry::types::triangleMesh_t & mesh_in
   );

   geometry::types::shape_t defaultShape(
      geometry::types::enumShape_t shape_type
   );

   // Returns true if the shape argument is polyhedral, false otherwise.
   bool polyhedralShape(geometry::types::shape_t shape);

   // Returns true if the shape enumeration is polyhedral, false otherwise.
   bool polyhedralShape(geometry::types::enumShape_t shape_type);

   geometry::types::triangleMesh_t loadShapeMesh(
      geometry::types::shape_t shape
   );

   geometry::types::triangleMesh_t loadDefaultShapeMesh(
      geometry::types::enumShape_t shape_type
   );

   data_triangleMesh_t loadDefaultShapeMeshData(
      geometry::types::enumShape_t shape_type, float scale
   );

   bool pointInsideAabb(
      const geometry::types::aabb_t & aabb,
      const Vector3 & p,
      float epsilon=1e-7f
   );

   // O(N) calculation of a point q's interiorness to a mesh.
   bool pointInsideMesh(
      const geometry::types::triangleMesh_t & mesh, const Vector3 & q
   );

   // O(N) calculation of a point q's interiorness to a mesh. Takes 'distance'
   // as an output arg, which is the distance from the closest point on the
   // mesh to the point q.
   bool pointInsideMesh(
      const geometry::types::triangleMesh_t & mesh,
      const Vector3 & q,
      float & distance
   );

   // Assumes that the ray is in the same coordinate frame as the AABB. Finds
   // the point of intersection between a ray and an AABB, excluding degenerate
   // and empty solutions (e.g. intersection with > 1 point is treated as not
   // intersection).
   bool rayIntersect(
      const geometry::types::aabb_t & aabb,
      const Vector3 & ray_start,
      const Vector3 & ray_unit,
      Vector3 & intersection
   );

   // Assumes that the ray is in the same coordinate frame as the mesh. Finds
   // the point of intersection between a ray and a mesh, excluding degenerate
   // and empty solutions (e.g. intersection with > 1 point is treated as not
   // intersection).
   bool rayIntersect(
      const geometry::types::triangleMesh_t & mesh,
      const Vector3 & ray_start,
      const Vector3 & ray_unit,
      Vector3 & intersection
   );

   // Finds the closest point on a mesh to a query point, q. The closest point,
   // the triangle containing the closest point, and the point + bary
   // coordinates are calculated from this function.
   pointOnMesh_t closestPointToPoint(
      const geometry::types::triangleMesh_t & mesh, const Vector3 & q
   );

}

}

#endif
