#ifndef MESH_TRIANGLE_GRAPH_HEADER
#define MESH_TRIANGLE_GRAPH_HEADER

#include "geometry_types.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace geometry
{

namespace mesh
{

   class TriangleGraphBase
   {
      public:
         // I want a static const variable inside of a templated class and I'm
         // going to have it by hook or by crook.
         // Each triangle's edge refers to specific indices for its vert IDs.
         // So if a triangle has vert IDs [2, 4, 5], these will be the edges:
         //    edge 0: [2, 4]
         //    edge 1: [4, 5]
         //    edge 2: [2, 5]
         // Which means that an edge is defined by the indices of its vert IDs.
         //    edge 0: vert ID indices [0, 1]
         //    edge 1: vert ID indices [1, 2]
         //    edge 2: vert ID indices [0, 2]
         static const unsigned int edgeVerts[3][2];
   };

   // This assumes that an intersection of two bodies results in a well-behaved
   // mesh (e.g. every triangle has exactly three neighboring triangles).
   // Each triangle has an array of 3 vertex IDs and an array of 3 neighbor
   // IDs. Vertex IDs correspond to positions of the vertices in a mesh array.
   // Vertex IDs are sorted in ascending order in the vertex ID array.
   // Neighbor IDs correspond to positions of triangles within the triangle
   // array in this class. The existence of a neighbor as well as its position
   // in the neighbor array depend on the existence of shared pairs of vertex
   // IDs between two triangles, and the positions of those two vertex IDs in
   // the vertex ID array.
   // Triangle A has a neighboring triangle B if A and B share two of the same
   // vertex IDs. Assuming A and B are neighbors, A marks B as a neighbor by
   // placing ID B into a specific slot in the neighbor array according to this
   // pattern:
   // 
   //    A.neighborIds[0] = B: (A.vertIds[0], A.vertIds[1]) are in B.vertIds
   //    A.neighborIds[1] = B: (A.vertIds[1], A.vertIds[2]) are in B.vertIds
   //    A.neighborIds[2] = B: (A.vertIds[0], A.vertIds[2]) are in B.vertIds
   // 
   // And similar for B. This means that the position of neighborship
   // indication between A and B may not be symmetrical, and that's OK.
   // A value of -1 in either the vertex ID array or the neighbor ID array
   // indicates that no vertex ID or neighbor ID has been assigned to that
   // slot.
   // Note that this formalism is adopted to avoid using DFS for finding
   // triangles that have had neighbors removed. The result is that, in this
   // naive implementation, finding open edges in an incomplete polytope is an
   // O(N) operation. This could be reduced dramatically by having a separate
   // array of pointers to keep track of "incomplete" triangles. Incomplete
   // meaning "having fewer than three neighbors".
   // Also note that these operations require no extra memory allocation beyond
   // the instantiation of this class.
   template <unsigned int MAX_SIZE>
   class TriangleGraphMesh : public TriangleGraphBase
   {
      typedef geometry::types::leanNeighborTriangle_t leanNeighborTriangle_t;

      public:
         TriangleGraphMesh(void)
         : free_index_(0)
         { }

         // Number of triangles in the array.
         unsigned int size(void) const
         {
            return free_index_;
         }

         // Returns true if there are no triangles in the array.
         bool empty(void) const
         {
            return free_index_ < MAX_SIZE;
         }

         // Returns true if the maximum number of triangles are in the array.
         bool full(void) const
         {
            return free_index_ >= MAX_SIZE - 1;
         }

         // Removes the triangle at index deleted_triangle_id. Adjusts all of
         // the neighbor linkages of any triangles at indices greater than
         // deleted_triangle_id. Any remaining triangles with the deleted
         // triangle as a neighbor will have that neighbor ID set to -1.
         void remove(int tri_id)
         {
            // Sever the connections from the deleted triangle's neighors to
            // the deleted triangle.
            changeNeighborConnections(tri_id, -1);

            if ((unsigned int )tri_id == free_index_ - 1)
            {
               free_index_ -= 1;
               return;
            }

            // Move the last triangle in the graph to the hole created by the
            // deleted triangle.
            unsigned int fill_tri_id = free_index_ - 1;

            // Update the hole filler's neighbors to refer to the location of
            // the deleted triangle's ID.
            changeNeighborConnections(fill_tri_id, tri_id);
            array_[tri_id] = array_[fill_tri_id];
            free_index_ -= 1;
         }

         // Similar to remove, but a copy of the removed triangle is returned.
         leanNeighborTriangle_t pop(unsigned int tri_id)
         {
            leanNeighborTriangle_t ret = array_[tri_id];
            remove(tri_id);
            return ret;
         }

         // Adds a fully-formed triangle to the array.
         int push_back(const leanNeighborTriangle_t & triangle)
         {
            if (full())
            {
               return -1;
            }

            // All vert IDs must be unique and must be in ascending order.
            if (
               !(
                  (triangle.vertIds[0] < triangle.vertIds[1]) &&
                  (triangle.vertIds[1] < triangle.vertIds[2])
               )
            )
            {
               return -1;
            }

            array_[free_index_] = triangle;
            free_index_ = std::min(
               free_index_ + 1, (unsigned int )(MAX_SIZE - 1)
            );

            return 1;
         }

         // Create a triangle out of potentially unordered vertex IDs and add
         // that triangle to the mesh.
         int add_triangle(int vert_id_a, int vert_id_b, int vert_id_c)
         {
            int temp_vert_ids[3] = {vert_id_a, vert_id_b, vert_id_c};
            std::sort<int *>(temp_vert_ids, temp_vert_ids + 3);

            leanNeighborTriangle_t temp_triangle = {
               {temp_vert_ids[0], temp_vert_ids[1], temp_vert_ids[2]},
               {-1, -1, -1}
            };

            int push_result = push_back(temp_triangle);
            if (push_result < 0)
            {
               return push_result;
            }

            unsigned int new_tri_id = free_index_ - 1;

            // Fill out the neighbors by looping over existing triangles and
            // seeing if any of them share edges with the new triangle.
            int temp_edge_a = -1;
            int temp_edge_b = -1;
            bool found_neighbor = false;
            unsigned int num_neighbors_found = 0;
            for (
               unsigned int old_tri_id = 0;
               old_tri_id < new_tri_id;
               ++old_tri_id
            )
            {
               findSharedEdge(old_tri_id, new_tri_id, temp_edge_a, temp_edge_b);
               found_neighbor = (temp_edge_a != -1 && temp_edge_b != -1);

               if (found_neighbor)
               {
                  addNeighbors(old_tri_id, temp_edge_a, new_tri_id, temp_edge_b);
               }

               num_neighbors_found += found_neighbor;
               if (num_neighbors_found == 3)
               {
                  break;
               }
            }

            return 1;
         }

         geometry::types::leanNeighborTriangle_t & operator[] (unsigned int i)
         {
            return array_[i];
         }

         const geometry::types::leanNeighborTriangle_t & operator[] (unsigned int i) const
         {
            return array_[i];
         }

         void clear(void)
         {
            free_index_ = 0;
         }

         void print(void) const
         {
            for (unsigned int i = 0; i < size(); ++i)
            {
               print(i);
            }
         }

         void print(unsigned int id) const
         {
            std::cout << "triangle id: " << id << "\n";
            std::cout << "\tvert ids: ";
            for (unsigned int i = 0; i < 3; ++i)
            {
               std::cout << array_[id].vertIds[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "\tneighbor ids: ";
            for (unsigned int i = 0; i < 3; ++i)
            {
               std::cout << array_[id].neighborIds[i] << " ";
            }
            std::cout << std::endl;
         }

      private:

         unsigned int free_index_;

         leanNeighborTriangle_t array_[MAX_SIZE];

         // Later functionality - keep track of the triangles that aren't
         // surrounded by neighbors and loop over these when adding new
         // triangles. When a triangle is removed, update the list of partially
         // connected triangles.
         // unsigned int partiallyConnectedTriangleIds[MAX_SIZE];

         // If a shared edge exists between the two triangles at tri_id_a and
         // tri_id_b, edge_a and edge_b are set to the vert IDs of that shared
         // edge.
         void findSharedEdge(
            unsigned int tri_id_a,
            unsigned int tri_id_b,
            int & edge_a,
            int & edge_b
         )
         {
            const leanNeighborTriangle_t & tri_a = array_[tri_id_a];
            const leanNeighborTriangle_t & tri_b = array_[tri_id_b];
            for (unsigned int i = 0; i < 3; ++i)
            {
               unsigned int vert_a0 = TriangleGraphMesh<MAX_SIZE>::edgeVerts[i][0];
               unsigned int vert_a1 = TriangleGraphMesh<MAX_SIZE>::edgeVerts[i][1];
               for (unsigned int j = 0; j < 3; ++j)
               {
                  unsigned int vert_b0 = TriangleGraphMesh<MAX_SIZE>::edgeVerts[j][0];
                  unsigned int vert_b1 = TriangleGraphMesh<MAX_SIZE>::edgeVerts[j][1];
                  if (
                     (tri_a.vertIds[vert_a0] != -1) &&
                     (tri_a.vertIds[vert_a1] != -1) &&
                     (tri_a.vertIds[vert_a0] == tri_b.vertIds[vert_b0]) &&
                     (tri_a.vertIds[vert_a1] == tri_b.vertIds[vert_b1]) &&
                     (tri_a.neighborIds[i] == -1) &&
                     (tri_b.neighborIds[j] == -1)
                  )
                  {
                     edge_a = i;
                     edge_b = j;
                     return;
                  }
               }
            }

            edge_a = -1;
            edge_b = -1;
         }

         // Update triangles with IDs tri_id_a and tri_id_b with neighbor
         // information. Triangle A shares edge A with triangle B, and triangle
         // B shares edge B with triangle A.
         void addNeighbors(
            unsigned int tri_id_a,
            unsigned int edge_a,
            unsigned int tri_id_b,
            unsigned int edge_b
         )
         {
            array_[tri_id_a].neighborIds[edge_a] = tri_id_b;
            array_[tri_id_b].neighborIds[edge_b] = tri_id_a;
         }

         // All of the neighbors of old_tri_id will have their connections to
         // old_tri_id changed to new_tri_id.
         // E.g.
         //    triangle 2 neighbors [ 3,  5,  6]
         //    triangle 3 neighbors [ 2,  4,  5]
         //    triangle 5 neighbors [ 0,  2,  8]
         //    triangle 6 neighbors [-1, 2,  6]
         // Calling `changeNeighborConnections(2, 15)` results in:
         //    triangle 2 neighbors [ 3,  5,  6]
         //    triangle 3 neighbors [ 15,  4,  5]
         //    triangle 5 neighbors [ 0,  15,  8]
         //    triangle 6 neighbors [-1, 15,  6]
         void changeNeighborConnections(int old_tri_id, int new_tri_id)
         {
            const leanNeighborTriangle_t & local = array_[old_tri_id];

            // Sever the connections from any triangles connected to the local
            // triangle.
            for (unsigned int i = 0; i < 3; ++i)
            {
               if (local.neighborIds[i] == -1)
               {
                  continue;
               }

               // Find the edge on the neighbor that has the local triangle
               // as a neighbor.
               leanNeighborTriangle_t & neighbor = array_[local.neighborIds[i]];
               for (unsigned int j = 0; j < 3; ++j)
               {
                  if (neighbor.neighborIds[j] == old_tri_id)
                  {
                     neighbor.neighborIds[j] = new_tri_id;
                     break;
                  }
               }
            }
         }
   };

}

}

#endif
