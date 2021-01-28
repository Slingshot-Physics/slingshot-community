set(
   headers
   include/aabb_hull.hpp
   include/aabb_tree.hpp
   include/epa.hpp
   include/epa_types.hpp
   include/gauss_map.hpp
   include/geometry_types.hpp
   include/geometry_type_converters.hpp
   include/gjk.hpp
   include/inertia.hpp
   include/line.hpp
   include/mesh.hpp
   include/mesh_ops.hpp
   include/plane.hpp
   include/polygon.hpp
   include/quickhull.hpp
   include/raycast.hpp
   include/sat.hpp
   include/segment.hpp
   include/shape_features.hpp
   include/support_functions.hpp
   include/tetrahedron.hpp
   include/transform.hpp
   include/triangle.hpp
   include/triangle_graph_mesh.hpp
   include/volume.hpp
   include/geometry.hpp
)

set(
   sources
   src/aabb_hull.cpp
   src/aabb_tree.cpp
   src/epa.cpp
   src/epa_types.cpp
   src/gauss_map.cpp
   src/gjk.cpp
   src/inertia.cpp
   src/geometry_type_converters.cpp
   src/line.cpp
   src/mesh.cpp
   src/mesh_ops.cpp
   src/plane.cpp
   src/polygon.cpp
   src/quickhull.cpp
   src/raycast.cpp
   src/sat.cpp
   src/segment.cpp
   src/shape_features.cpp
   src/support_functions.cpp
   src/tetrahedron.cpp
   src/transform.cpp
   src/triangle_graph_mesh.cpp
   src/triangle.cpp
   src/volume.cpp
)

set(
   target
   edbdgeometry
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
   ${target}
   datamodel
   edbdgraph
   edbdlogger
   edbdmath
)

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -Werror)
