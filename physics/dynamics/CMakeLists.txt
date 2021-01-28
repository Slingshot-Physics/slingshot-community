set(
    headers
    generalized_force_reset_system.hpp
    rigidbody.hpp
    rk4_finalizer_system.hpp
    rk4_increment_calculator_system.hpp
    rk4_integrator.hpp
    rk4_integrator_system.hpp
    rk4_midpoint_calculator_system.hpp
)

set(
    sources
    generalized_force_reset_system.cpp
    rigidbody.cpp
    rk4_finalizer_system.cpp
    rk4_integrator.cpp
    rk4_integrator_system.cpp
)

set(
    target
    bodydynamics
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
    datamodel
    edbdmath
    slingshottypes
    trecs
)
