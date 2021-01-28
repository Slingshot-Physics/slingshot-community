set(
   headers
   collision/collision.hpp
   friction/friction.hpp
   gear/gear.hpp
   revolute_motor/revolute_motor.hpp
   rotation_1d/rotation_1d.hpp
   torsional_friction/torsional_friction.hpp
   translation_1d/translation_1d.hpp
   collision_system.hpp
   friction_system.hpp
   torsional_friction_system.hpp
   constrained_rigidbody_system.hpp
   constraint_output.hpp
   constraint_solver.hpp
   constraint_stove.hpp
   equality_constraint_calculator_system.hpp
   inequality_constraint_calculator_system.hpp
)

set(
   sources
   collision/collision.cpp
   friction/friction.cpp
   gear/gear.cpp
   revolute_motor/revolute_motor.cpp
   rotation_1d/rotation_1d.cpp
   torsional_friction/torsional_friction.cpp
   translation_1d/translation_1d.cpp
   constrained_rigidbody_system.cpp
   constraint_output.cpp
   constraint_solver.cpp
   constraint_stove.cpp
)

set(
   target
   constraint_pipeline
)

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
   ${CMAKE_CURRENT_SOURCE_DIR}/balljoint
   ${CMAKE_CURRENT_SOURCE_DIR}/collision
   ${CMAKE_CURRENT_SOURCE_DIR}/friction
   ${CMAKE_CURRENT_SOURCE_DIR}/gear
   ${CMAKE_CURRENT_SOURCE_DIR}/revolute_joint
   ${CMAKE_CURRENT_SOURCE_DIR}/revolute_motor
   ${CMAKE_CURRENT_SOURCE_DIR}/rotation_1d
   ${CMAKE_CURRENT_SOURCE_DIR}/torsional_friction
   ${CMAKE_CURRENT_SOURCE_DIR}/translation_1d
)

target_link_libraries(
   ${target}
   bodydynamics
   datamodel
   edbdgraph
   edbdlogger
   edbdmath
   slingshottypes
   trecs
)
