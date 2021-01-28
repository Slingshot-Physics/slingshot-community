set(
   headers
   constant_force.hpp
   constant_force_calculator.hpp
   drag_force.hpp
   drag_force_calculator.hpp
   drag_torque.hpp
   drag_torque_calculator.hpp
   spring_force_calculator.hpp
   spring_force.hpp
   velocity_damper_force_calculator.hpp
   velocity_damper_force.hpp
)

set(
   sources
   constant_force.cpp
   constant_force_calculator.cpp
   drag_force.cpp
   drag_force_calculator.cpp
   drag_torque.cpp
   drag_torque_calculator.cpp
   spring_force_calculator.cpp
   spring_force.cpp
   velocity_damper_force_calculator.cpp
   velocity_damper_force.cpp
)

set(
   target
   forces
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

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
   ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
   ${target}
   bodydynamics
   datamodel
   edbdmath
   slingshottypes
   trecs
)
