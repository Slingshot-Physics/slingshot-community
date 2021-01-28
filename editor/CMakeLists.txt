set(
   target
   editor
)

set(
   headers
   camera_control_gui.hpp
   editor.hpp
   editor_gui.hpp
   widgets/component_widget_interface.hpp
   widgets/balljoint_widget.hpp
   widgets/component_widget_interface.hpp
   widgets/constant_force_widget.hpp
   widgets/damper_force_widget.hpp
   widgets/drag_torque_widget.hpp
   widgets/drag_force_widget.hpp
   widgets/gear_widget.hpp
   widgets/revolute_joint_widget.hpp
   widgets/revolute_motor_widget.hpp
   widgets/rigid_body_widget.hpp
   widgets/rotation_1d_widget.hpp
   widgets/shape_widgets.hpp
   widgets/spring_force_widget.hpp
   widgets/translation_1d_widget.hpp
)

set(
   sources
   camera_control_gui.cpp
   editor.cpp
   editor_gui.cpp
   main.cpp
   widgets/component_widget_interface.cpp
   widgets/balljoint_widget.cpp
   widgets/constant_force_widget.cpp
   widgets/damper_force_widget.cpp
   widgets/drag_torque_widget.cpp
   widgets/drag_force_widget.cpp
   widgets/gear_widget.cpp
   widgets/revolute_joint_widget.cpp
   widgets/revolute_motor_widget.cpp
   widgets/rigid_body_widget.cpp
   widgets/rotation_1d_widget.cpp
   widgets/shape_widgets.cpp
   widgets/spring_force_widget.cpp
   widgets/translation_1d_widget.cpp
)

add_executable(
   ${target}
   ${sources}
   ${headers}
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -Werror)

target_link_libraries(
   ${target}
   slingshot
)
