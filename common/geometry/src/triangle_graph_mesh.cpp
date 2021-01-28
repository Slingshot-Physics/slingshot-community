#include "triangle_graph_mesh.hpp"

#include <iostream>

namespace geometry
{

namespace mesh
{
   // Initialize the static member of the TriangleGraph 
   const unsigned int TriangleGraphBase::edgeVerts[3][2] = {
      {0, 1}, {1, 2}, {0, 2}
   };

}

}
