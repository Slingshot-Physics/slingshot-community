set(
   headers
   include/dynamic_array.hpp
)

set(
   sources
)

set(
   target
   edbdcontainers
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

set_target_properties(
   ${target}
   PROPERTIES
   LINKER_LANGUAGE
   CXX
)

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -Werror)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/include
)
