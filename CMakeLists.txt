cmake_minimum_required( VERSION 3.16 )

set(CMAKE_CXX_STANDARD 11)

project(
    SLINGSHOT
    VERSION 0.0.1
)

include( ${CMAKE_SOURCE_DIR}/cmake/functions.cmake )

option(BUILD_TESTS "Build tests with the physics engine" ON)
option(BUILD_VIZ "Build the visualization library with the physics engine" ON)
option(BUILD_CUDA "Build any cuda-enabled code with nvcc" OFF)
option(BUILD_SSE "Build the engine with SSE enabled" ON)

# Command-line options
if (DEFINED HELP_SOS)
    message(STATUS "")
    message(STATUS "Custom flags!")
    message(STATUS "    -DCMAKE_BUILD_TYPE=[debug, release]")
    message(STATUS "    -DOPTIMIZATION_OPT=[o0, o1, o2, o3]")
    message(STATUS "")
    return()
endif()

if (NOT DEFINED CMAKE_BUILD_TYPE)
    message(STATUS "adding debug flags")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
elseif (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "Building for: ${CMAKE_BUILD_TYPE}" )
    set(CMAKE_BUILD_TYPE debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
endif()

if (NOT DEFINED OPTIMIZATION_OPT)
    set(OPTIMIZATION_OPT "O0")
endif()
string(TOLOWER ${OPTIMIZATION_OPT} OPTIMIZATION_OPT)

message(STATUS "Building with optimization: ${OPTIMIZATION_OPT}")

# Default is no optimization
if (${OPTIMIZATION_OPT} STREQUAL "o1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
elseif (${OPTIMIZATION_OPT} STREQUAL "o2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
elseif (${OPTIMIZATION_OPT} STREQUAL "o3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

add_compile_definitions(
    SLINGSHOT
    PRIVATE
    SOURCE_DIR="${CMAKE_SOURCE_DIR}/.."
    BIN_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

if ( BUILD_SSE )
    add_compile_definitions(
        SLINGSHOT
        PRIVATE
        BUILD_SSE
    )
endif()

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

# How to copy libraries (DLL's on Windows) over to binary directory
# https://stackoverflow.com/a/63158941

# Adding compile definitions here makes them available in *all* source files.
if ( BUILD_VIZ )
   message( STATUS "Adding build_viz flag to entire repo!")
   add_compile_definitions( BUILD_VIZ )
endif()

add_subdirectory( common )
add_subdirectory( datamodel )
add_subdirectory( demos )
add_subdirectory( physics )
add_subdirectory( slingshot )

if ( BUILD_VIZ )
    message( STATUS "building viz library" )
    add_subdirectory( viz )
    add_subdirectory( editor )
endif()

if ( BUILD_CUDA )
    add_subdirectory( cuda_common )
endif()

if ( BUILD_TESTS )
    message( STATUS "building tests" )
    add_subdirectory( tests )
endif()
