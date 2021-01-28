#ifndef QUICKHULL_HEADER
#define QUICKHULL_HEADER

#include "geometry_types.hpp"
#include "triangle_graph_mesh.hpp"
#include "vector3.hpp"

namespace geometry
{

namespace mesh
{

   typedef geometry::types::minkowskiDiffVertex_t md_vert_t;

   struct quickhullMesh_t
   {
      geometry::mesh::TriangleGraphMesh<MAX_VERTICES> triangles;

      unsigned int numVerts;
      md_vert_t verts[MAX_VERTICES];

      // If successful, returns the index of the newly added vertex.
      // Returns -1 if unsuccessful.
      int addVert(const md_vert_t & md_vert);

   };

   // Copies vertices and triangles from the quick hull mesh to a standard
   // mesh.
   void convertQuickHullToMeshConfig(
      const quickhullMesh_t & qhull,
      const Vector3 & hull_center,
      geometry::types::triangleMesh_t & convex_hull
   );

   // Generates an initial tetrahedron out of the array of points.
   void findTetrahedron(
      unsigned int num_points,
      const md_vert_t * points,
      quickhullMesh_t & qhull,
      bool * verts_added
   );

   // Finds the index of the farthest exterior non-hull point from the closest
   // triangle on the hull. Returns the index of the farthest exterior point.
   // Returns -1 if no farthest index can be found, indicating that there are
   // no points that can be added to the hull.
   int findFarthestPoint(
      const quickhullMesh_t & qhull,
      const Vector3 & hull_center,
      const bool * verts_added,
      unsigned int num_points,
      const md_vert_t * points
   );

   // Finds and deletes all triangles on the quickhull mesh that have the
   // query point in their positive half spaces.
   void deleteTrianglesFacingPoint(
      const md_vert_t & point, const Vector3 & hull_center, quickhullMesh_t & qhull
   );

   // Finds any triangles in the quickhull mesh with unneighbored edges and
   // creates new triangles from those edges to ID of the newest vertex. The
   // ID of the newest vertex should be the index of the last vertex added to
   // the quickhull mesh.
   int addTrianglesToVertId(unsigned int new_vert_id, quickhullMesh_t & qhull);

   // Calculates the convex hull of an array of labeled Minkowski Difference
   // vertices. Returns a standard mesh as an output arg. A hull is only made
   // if the points are non-coplanar.
   // Returns 1 if successful. Returns -1 if unsuccessful.
   int generateHull(
      unsigned int num_points,
      const md_vert_t * points,
      geometry::types::triangleMesh_t & mesh_out
   );

   // Calculates the convex hull of an array of vertices. A hull is only made
   // if the points are non-coplanar.
   int generateHull(
      unsigned int num_points,
      const Vector3 * points,
      geometry::types::triangleMesh_t & mesh_out
   );

}

}

#endif
