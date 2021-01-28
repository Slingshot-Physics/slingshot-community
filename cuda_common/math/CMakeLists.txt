set(
   headers
   include/matrix33.cuh
   include/vector3.cuh
)

set(
   sources
   src/matrix33.cu
   src/vector3.cu
)

set(
   target
   edbdcudamath
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

# The nvcc linker doesn't allow linking between shared libraries if there is
# a mixture of device and host code between those libraries. Understandable.
add_library(
   ${target}
   STATIC
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
