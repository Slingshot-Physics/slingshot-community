add_subdirectory( dynamics )
add_subdirectory( collision_pipeline )
add_subdirectory( constraints )
add_subdirectory( forces )
add_subdirectory( types )

set(
    headers
    handle.hpp
)

set(
    sources
    handle.cpp
)

set(
    target
    physics
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

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unused-function -Werror)

target_link_libraries(
   ${target}
   bodydynamics
   collision_pipeline
   constraint_pipeline
   forces
   datamodel
   edbdlogger
   slingshottypes
   trecs
)
