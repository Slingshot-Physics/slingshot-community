set(
   headers
   include/geometry_types.cuh
   include/plane.cuh
   include/segment.cuh
   include/simplex.cuh
   include/tetrahedron.cuh
   include/triangle.cuh
)

set(
   sources
   src/plane.cu
   src/segment.cu
   src/simplex.cu
   src/tetrahedron.cu
   src/triangle.cu
)

set(
   target
   edbdcudageometry
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

target_link_libraries(
   ${target}
   edbdcudamath
)
