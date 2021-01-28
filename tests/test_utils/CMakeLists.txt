set(
   target
   test_utils
)

set(
   sources
   random_gjk_input_util.cpp
   random_graph_utils.cpp
   transform_utils.cpp
   triangle_utils.cpp
)

set(
   headers
   blank_systems.hpp
   md_hull_intersection.hpp
   random_gjk_input_util.hpp
   random_graph_utils.hpp
   test_input_serializer_util.hpp
   transform_utils.hpp
   triangle_utils.hpp
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
   ${target}
   edbdgeometry
   edbdgraph
   edbdmath
   trecs
)

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
