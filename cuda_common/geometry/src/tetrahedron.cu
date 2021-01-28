#include "tetrahedron.cuh"

#include "matrix33.cuh"
#include "triangle.cuh"

namespace cugeom
{

namespace tetrahedron
{

   __constant__ unsigned int num_tetra_edges = 6;

   __constant__ unsigned int num_tetra_faces = 4;

   __constant__ edge_t tetra_edge_combos[6] = {
      {0, 1, 2, 3},
      {0, 2, 1, 3},
      {0, 3, 1, 2},
      {1, 2, 0, 3},
      {1, 3, 0, 2},
      {2, 3, 0, 1}
   };

   __constant__ face_t tetra_face_combos[4] = {
      {0, 1, 2, 3},
      {0, 3, 1, 2},
      {1, 2, 3, 0},
      {2, 3, 0, 1}
   };

   CUDA_CALLABLE pointBaryCoord_t closestPointToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & d,
      const cumath::Vector3 & q
   )
   {
      pointBaryCoord_t bary_point;
      bary_point.point = q;

      // See wikipedia entry on Barycentric coordinates for tetrahedra.
      cumath::Vector3 vda = a - d;
      cumath::Vector3 vdb = b - d;
      cumath::Vector3 vdc = c - d;
      cumath::Matrix33 T(
         vda[0], vdb[0], vdc[0],
         vda[1], vdb[1], vdc[1],
         vda[2], vdb[2], vdc[2]
      );
      bary_point.baryCoords = (~T) * (q - d);

      return bary_point;
   }

   // Selects the Voronoi region that the point q belongs to from the
   // tetrahedron defined by [a, b, c, d].
   CUDA_CALLABLE simplex3_t voronoiRegionToPoint(
      const cumath::Vector3 & a,
      const cumath::Vector3 & b,
      const cumath::Vector3 & c,
      const cumath::Vector3 & d,
      const cumath::Vector3 & q
   )
   {
      cumath::Vector3 tetrahedron[4] = {a, b, c, d};
      cumath::Vector3 cm((a + b + c + d)/4.f);

      simplex3_t vertex_region;
      vertex_region.numVerts = 0;

      simplex3_t edge_region;
      edge_region.numVerts = 0;

      simplex3_t triangle_region;
      triangle_region.numVerts = 0;

      simplex3_t tetra_region;
      tetra_region.numVerts = 0;
      tetra_region.vertIdxs[0] = 0;
      tetra_region.vertIdxs[1] = 1;
      tetra_region.vertIdxs[2] = 2;
      tetra_region.vertIdxs[3] = 3;
      tetra_region.verts[0] = a;
      tetra_region.verts[1] = b;
      tetra_region.verts[2] = c;
      tetra_region.verts[3] = d;

      const int num_verts = 4;
      for (int i = 0; i < num_verts; ++i)
      {
         cumath::Vector3 pmq(q - tetrahedron[i]);
         bool ineq1 = \
            pmq.dot(tetrahedron[(i + 1) % num_verts] - tetrahedron[i]) <= 0.0f;
         bool ineq2 = \
            pmq.dot(tetrahedron[(i + 2) % num_verts] - tetrahedron[i]) <= 0.0f;
         bool ineq3 = \
            pmq.dot(tetrahedron[(i + 3) % num_verts] - tetrahedron[i]) <= 0.0f;

         if (ineq1 && ineq2 && ineq3)
         {
            vertex_region.numVerts = 1;
            vertex_region.vertIdxs[0] = i;
            vertex_region.verts[0] = tetrahedron[i];
         }
      }

      // Check edge feature regions.
      for (int i = 0; i < num_tetra_edges; ++i)
      {
         cumath::Vector3 & vert_a = tetrahedron[tetra_edge_combos[i].vertAId];
         cumath::Vector3 & vert_b = tetrahedron[tetra_edge_combos[i].vertBId];
         cumath::Vector3 & vert_c = tetrahedron[tetra_edge_combos[i].vertCId];
         cumath::Vector3 & vert_d = tetrahedron[tetra_edge_combos[i].vertDId];

         cumath::Vector3 nABC = (vert_b - vert_a).crossProduct(vert_c - vert_a);

         cumath::Vector3 nADB = (vert_d - vert_a).crossProduct(vert_b - vert_a);

         cumath::Vector3 pmq = (q - vert_a);
         cumath::Vector3 pmqp1 = (q - vert_b);
         bool ineq1 = pmq.dot(vert_b - vert_a) >= 1e-7f;
         bool ineq2 = pmqp1.dot(vert_a - vert_b) >= 1e-7f;
         bool ineq3 = pmq.dot((vert_b - vert_a).crossProduct(nABC)) >= 1e-7f;
         bool ineq4 = -1.0f * pmq.dot((vert_b - vert_a).crossProduct(nADB)) >= 1e-7f;

         if (ineq1 && ineq2 && ineq3 && ineq4)
         {
            edge_region.numVerts = 2;
            edge_region.vertIdxs[0] = tetra_edge_combos[i].vertAId;
            edge_region.vertIdxs[1] = tetra_edge_combos[i].vertBId;
            edge_region.verts[0] = vert_a;
            edge_region.verts[1] = vert_b;
         }
      }

      // Check triangle feature regions.
      for (int i = 0; i < num_tetra_faces; ++i)
      {
         cumath::Vector3 & vert_a = tetrahedron[tetra_face_combos[i].vertAId];
         cumath::Vector3 & vert_b = tetrahedron[tetra_face_combos[i].vertBId];
         cumath::Vector3 & vert_c = tetrahedron[tetra_face_combos[i].vertCId];
         cumath::Vector3 & vert_d = tetrahedron[tetra_face_combos[i].vertDId];

         cumath::Vector3 nABC = cugeom::triangle::calcNormal(vert_a, vert_b, vert_c, cm);

         cumath::Vector3 NABC = (vert_b - vert_a).crossProduct(vert_c - vert_a);
         cumath::Vector3 nnAB = (vert_b - vert_a).crossProduct(NABC);
         cumath::Vector3 nnBC = (vert_c - vert_b).crossProduct(NABC);
         cumath::Vector3 nnCA = (vert_a - vert_c).crossProduct(NABC);

         bool ineq1 = (q - vert_a).dot(nnAB) <= 0.0f;
         bool ineq2 = (q - vert_b).dot(nnBC) <= 0.0f;
         bool ineq3 = (q - vert_c).dot(nnCA) <= 0.0f;
         bool ineq4 = ((q - vert_a).dot(nABC) * (vert_d - vert_a).dot(nABC)) <= 0.0f;

         if (ineq1 && ineq2 && ineq3 && ineq4)
         {
            triangle_region.numVerts = 3;
            triangle_region.vertIdxs[0] = tetra_face_combos[i].vertAId;
            triangle_region.vertIdxs[1] = tetra_face_combos[i].vertBId;
            triangle_region.vertIdxs[2] = tetra_face_combos[i].vertCId;
            triangle_region.verts[0] = vert_a;
            triangle_region.verts[1] = vert_b;
            triangle_region.verts[2] = vert_c;
         }
      }

      return (
         (vertex_region.numVerts == 1) ? vertex_region : (
            (edge_region.numVerts == 2) ? edge_region : (
               (triangle_region.numVerts == 3) ? triangle_region : (
                  tetra_region
               )
            )
         )
      );

   }

}

}
