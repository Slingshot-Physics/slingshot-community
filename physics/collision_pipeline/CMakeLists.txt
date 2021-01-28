set(
    headers
    aabb_calculator.hpp
    broadphase_system.hpp
    collision_contacts.hpp
    collision_geometry_system.hpp
    collision_manifold_system.hpp
    manifold_component_jankifier.hpp
    narrowphase_system.hpp
    sat_system.hpp
)

set(
    sources
    collision_contacts.cpp
)

set(
    target
    collision_pipeline
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    ${target}
    edbdcontainers
    edbdgeometry
    edbdmath
    slingshottypes
    trecs
)
