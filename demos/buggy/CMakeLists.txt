set(
  sources
  viz.cpp
  buggy_callback.cpp
  buggy_camera_controller.cpp
)

set(
  headers
  buggy_callback.hpp
  buggy_camera_controller.hpp
)

set(
  target
  buggy_viz
)

add_executable(
  ${target}
  ${sources}
  ${headers}
)

target_link_libraries(
  ${target}
  slingshot
)
