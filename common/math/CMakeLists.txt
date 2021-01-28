set(
   headers
   include/attitudeutils.hpp
   include/heap_matrix.hpp
   include/matrix.hpp
   include/matrix33.hpp
   include/quadratic.hpp
   include/quaternion.hpp
   include/random_utils.hpp
   include/vector3.hpp
   include/vector4.hpp
)

set(
   sources
   src/attitudeutils.cpp
   src/heap_matrix.cpp
   src/matrix.cpp
   src/matrix33.cpp
   src/quadratic.cpp
   src/quaternion.cpp
   src/random_utils.cpp
   src/vector3.cpp
   src/vector4.cpp
)

if ( BUILD_SSE )
   list( APPEND headers include/simd_ops.hpp)
   list( APPEND sources src/simd_ops.cpp)
endif()

set(
   target
   edbdmath
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
