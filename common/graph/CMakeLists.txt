set(
   headers
   include/graph_types.hpp
   include/graph.hpp
   include/subgraph.hpp
   include/nary_tree.hpp
)

set(
   sources
   src/subgraph.cpp
)

set(
   target
   edbdgraph
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -Werror)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
   ${target}
   edbdcontainers
)

set_target_properties(
   ${target}
   PROPERTIES
   LINKER_LANGUAGE
   CXX
)
