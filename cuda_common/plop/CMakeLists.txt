set(
   headers
   include/host_array.cuh
   include/plop.cuh
)

set(
   sources
   src/plop.cu
)

set(
   target
   plop
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

# This can be shared because there's not actually any source code to link
# against, it's just header files.
add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

set_target_properties(
   ${target}
   PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/include
)
