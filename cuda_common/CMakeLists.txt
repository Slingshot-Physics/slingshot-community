cmake_minimum_required( VERSION 3.17 )

set(CMAKE_CXX_STANDARD 11)

project(
   CUDA_COMMON
   VERSION 0.0.1
   LANGUAGES CUDA CXX
)

# This must be 60. I don't know why, but it must. This will not build if
# CMAKE_CUDA_ARCHITECTURES is 70.
if ( NOT DEFINED CMAKE_CUDA_ARCHITECTURES )
   set( CMAKE_CUDA_ARCHITECTURES 60 )
endif()

include( CheckLanguage )

check_language( CUDA )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_subdirectory( buffers )
add_subdirectory( geometry )
add_subdirectory( math )
add_subdirectory( plop )
add_subdirectory( tests )
