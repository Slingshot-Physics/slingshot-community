set(
  sources
  main.cpp
  simple_callback.cpp
  point_orbit_camera_controller.cpp
  point_orbit_gui.cpp
)

set(
  headers
  simple_callback.hpp
  point_orbit_camera_controller.hpp
  point_orbit_gui.hpp
)

set(
  target
  orbit_viz
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
