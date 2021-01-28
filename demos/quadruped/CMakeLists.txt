set(
  sources
  main.cpp
  quadruped_callback.cpp
  quadruped_gui.cpp
)

set(
  headers
  quadruped_callback.hpp
  quadruped_gui.hpp
)

set(
  target
  quadruped
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
