#ifndef GEOMETRY_TYPES_HEADER
#define GEOMETRY_TYPES_HEADER

#include "matrix33.hpp"
#include "vector3.hpp"
#include "vector4.hpp"

#define MAX_TRIANGLES 1000
#define MAX_VERTICES 500

namespace geometry
{

namespace types
{
   enum class enumShape_t
   {
      CUBE = 0,
      CYLINDER = 1,
      SPHERE = 4,
      CAPSULE = 7,
      NONE = 404,
   };

   // Assuming body center is at (0, 0, 0) in body frame:
   //    x' = rotate * scale * x + translate
   struct transform_t
   {
      // Scale along arbitrary axes in body frame.
      Matrix33 scale;

      // Rotation matrix from body frame to world frame.
      Matrix33 rotate;

      // The position of the body's CM in world frame.
      Vector3 translate;
   };

   // Assuming body center is at (0, 0, 0) in body frame:
   //    x' = rotate * x + translate
   struct isometricTransform_t
   {
      // Rotation matrix from body frame to world frame.
      Matrix33 rotate;

      // The position of the body's CM in world frame.
      Vector3 translate;
   };

   struct aabb_t
   {
      Vector3 vertMax;
      Vector3 vertMin;
   };

   struct meshTriangle_t
   {
      unsigned int vertIds[3];
      Vector3 normal;
   };

   struct polygon50_t
   {
      unsigned int numVerts;
      Vector3 verts[50];
   };

   struct polygon4_t
   {
      unsigned int numVerts;
      Vector3 verts[4];
   };

   struct pointBaryCoord_t
   {
      // The actual point on a simplex.
      Vector3 point;
      // The barycentric coordinates of the point on a 1-, 2-, or 3-simplex.
      Vector4 bary;
   };

   struct clippedSegment_t
   {
      bool clipped;
      Vector3 points[2];
   };

   struct segment_t
   {
      Vector3 points[2];
   };

   struct segmentClosestPoints_t
   {
      unsigned int numPairs;
      Vector3 segmentPoints[2];
      Vector3 otherPoints[2];
   };

   struct plane_t
   {
      Vector3 point;
      Vector3 normal;
   };

   struct triangle_t
   {
      Vector3 verts[3];
   };

   struct tetrahedron_t
   {
      Vector3 verts[4];
   };

   struct leanNeighborTriangle_t
   {
      // Indices of the vertices in the mesh contributing to this triangle.
      int vertIds[3];
      // Indices/IDs of the three triangles sharing sides with this triangle.
      // Refers to triangles in the MeshTriangleGraph class.
      int neighborIds[3];
      // Could include a cached normal vector to make positive half-space
      // calculations faster (since the triangle deletion process is O(N)).
   };

   struct triangleMesh_t
   {
      unsigned int numTriangles;
      unsigned int numVerts;
      meshTriangle_t triangles[MAX_TRIANGLES];
      Vector3 verts[MAX_VERTICES];
   };

   // For use only with gaussMapMesh types. The normal is a face normal on a
   // convex polyhedron. The triangleStartId contains the first index of the
   // triangles that are on the face of the Gauss map. The numTriangles is the
   // number of triangles from the mesh that are on this face. All triangle
   // indices from triangleStartId to triangleStartId + numTriangles - 1 are
   // part of this face.
   struct gaussMapFace_t
   {
      // The normal of the face.
      Vector3 normal;
      // The first index of the triangles that comprise this face.
      unsigned int triangleStartId;
      // The total number of triangles on this face.
      unsigned int numTriangles;
   };

   // A mesh definition where triangles are neighbored and faces are first-
   // class citizens. Each face contains a normal and a range of indices of
   // triangles in the 'triangles' array that make up that face.
   // The elements of 'triangles' are ordered so that sequential ranges of
   // triangle indices belong to one face.
   // Functions using this data type will expect special orderings of triangle
   // vertices.
   struct gaussMapMesh_t
   {
      // The number of neighbored triangles in this mesh.
      unsigned int numTriangles;
      // The number of vertices in this mesh.
      unsigned int numVerts;
      // The number of faces in this mesh.
      unsigned int numFaces;
      // The neighborhood of triangles in the mesh. The triangles should be in
      // an order that allows ranges of triangle IDs to be on a face.
      leanNeighborTriangle_t triangles[MAX_TRIANGLES];
      // The mesh vertices. There is no special ordering of vertices.
      Vector3 verts[MAX_VERTICES];
      // The faces of the mesh. There is no special ordering of faces.
      gaussMapFace_t faces[MAX_TRIANGLES];
   };

   // A vertex in a Minkowski difference contains the IDs of the vertices on
   // the meshes that generated the actual MD vert. It also contains the points
   // on the surfaces that generated the actual MD vert (for implicit shapes).
   struct minkowskiDiffVertex_t
   {
      // The index number of the vertices in body A that contribute to the
      // simplex vertex.
      int bodyAVertId;

      // The index number of the vertices in body A that contribute to the
      // simplex vertex.
      int bodyBVertId;

      // The Minkowski-difference vertex.
      Vector3 vert;

      // The point on body A's surface that contributes to 'vert'.
      Vector3 bodyAVert;

      // The point on body B's surface that contributes to 'vert'.
      Vector3 bodyBVert;
   };

   // This Minkowski difference simplex contains a set of vertices on the
   // perimeter of the Minkowski difference between two convex shapes.
   struct minkowskiDiffSimplex_t
   {
      // Refers to the index numbers of the vertices in body A that contribute
      // to the simplex vertex at the same index.
      //    E.g. bodyAVertIds[0] contributes to verts[0]
      int bodyAVertIds[4];

      // Refers to the index numbers of the vertices in body B that contribute
      // to the simplex vertex at the same index.
      //    E.g. bodyAVertIds[0] contributes to verts[0]
      int bodyBVertIds[4];

      // Barycentric coordinates of the point on the simplex that's closest to
      // the origin.
      Vector4 minNormBary;

      // The number of active vertices in the simplex.
      unsigned int numVerts;

      // The vertices of the simplex in the coordinate system specified by the
      // GJK implementation.
      Vector3 verts[4];

      // The vertices of bodies A that make up the vertices on the simplex.
      Vector3 bodyAVerts[4];

      // The vertices of bodies B that make up the vertices on the simplex.
      Vector3 bodyBVerts[4];
   };

   // Array of vertices in a convex polyhedron with a center point.
   struct convexPolyhedron_t
   {
      unsigned int numVerts;
      Vector3 verts[MAX_VERTICES];
      Vector3 center;
   };

   // A 3D vertex with an integer label.
   struct labeledVertex_t
   {
      // Vertex ID in the Minkowski difference that yields the support
      // point. Only valid for convex polyhedra.
      int vertId;
      Vector3 vert;
   };

   // The output of GJK bundled into one structure.
   struct gjkResult_t
   {
      // The final minimum simplex used by GJK to test for intersection.
      minkowskiDiffSimplex_t minSimplex;

      // True if an intersection exists, false otherwise.
      bool intersection;
   };

   struct epaResult_t
   {
      // Boo, this is a hack. Decide to disallow the collision if EPA can't
      // find a decent vector of penetration.
      bool collided;

      // Vector of maximum penetration from body A to body B (world coords).
      Vector3 p;

      // One of the points on body A that penetrates most deeply into body B
      // in world coordinates. Points of deepest contact may be degenerate.
      Vector3 bodyAContactPoint;

      // One of the points on body B that penetrates most deeply into body A
      // in world coordinates. Points of deepest contact may be degenerate.
      Vector3 bodyBContactPoint;
   };

   struct satResult_t
   {
      // True if collision occurs, false otherwise.
      bool collision;

      // Contact direction from body A to body B if the bodies are colliding,
      // or the axis of separation if the bodies are not colliding.
      Vector3 contactNormal;

      unsigned int numDeepestPointPairs;

      // Points on A that are closest or deepest into body B.
      // If the bodies are colliding, deepestPointsA[i] and deepestPointsB[i]
      // are penetration pairs.
      // If the bodies are not colliding, deepestPointsA[i] is closest to
      // deepestPointsB[i].
      // Deepest points are extremal points in the positive or negative
      // direction of the contact normal or separation axis.
      Vector3 deepestPointsA[4];

      // Points on B that are closest or deepest into body A.
      // If the bodies are colliding, deepestPointsA[i] and deepestPointsB[i]
      // are penetration pairs.
      // If the bodies are not colliding, deepestPointsA[i] is closest to
      // deepestPointsB[i].
      // Deepest points are extremal points in the positive or negative
      // direction of the contact normal or separation axis.
      Vector3 deepestPointsB[4];
   };

   // A polyhedron feature is a locus of points on the surface of a convex
   // shape whose normals are all aligned.
   // A polyhedron can have face, edge, or vertex features.
   struct polyhedronFeature_t
   {
      polygon50_t shape;
      Vector3 normal;
   };

   struct shapeCube_t
   {
      // x-axis
      float length;

      // y-axis
      float width;

      // z-axis
      float height;
   };

   // Capsule centered on the origin with control points on the +/-z axis in
   // body frame.
   struct shapeCapsule_t
   {
      // Radius of the spheres on either end of the capsule
      float radius;

      // Distance between the control points on the +/-z axis in body frame
      float height;
   };

   // Cylinder whose center of geometry is on the origin.
   struct shapeCylinder_t
   {
      // Radius of the shell about the symmetry axis
      float radius;

      // Height of the cylinder
      float height;
   };

   // Sphere centered at the origin.
   struct shapeSphere_t
   {
      float radius;
   };

   struct shape_t
   {
      // Enumeration indicating which element of the union should be used.
      enumShape_t shapeType;

      // Anonymous union to clean up accessor syntax.
      union
      {
         shapeCapsule_t capsule;
         shapeCube_t cube;
         shapeCylinder_t cylinder;
         shapeSphere_t sphere;
      };
   };

   struct raycastResult_t
   {
      bool hit;
      unsigned int numHits;
      Vector3 hits[2];

      // Parameters generating the near and far hits out of the start and end
      // points on the ray. If there are two hits:
      //    hits[0] = ray_start * u[0] + ray_end
      //    hits[1] = ray_start * u[1] + ray_end
      float u[2];
   };

   struct testGjkInput_t
   {
      geometry::types::transform_t transformA;
      geometry::types::shape_t shapeA;
      geometry::types::transform_t transformB;
      geometry::types::shape_t shapeB;
   };

   struct testTetrahedronInput_t
   {
      geometry::types::tetrahedron_t tetrahedron;
      Vector4 queryPointBary;
      Vector3 queryPoint;
   };

   struct testTriangleInput_t
   {
      geometry::types::triangle_t triangle;
      Vector3 queryPointBary;
      Vector3 queryPoint;
   };

}

}

#endif
