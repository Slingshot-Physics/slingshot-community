set(
  sources
  main.cpp
  ld52_callback.cpp
  ld52_camera_controller.cpp
  ld52_gui.cpp
  veggie_state.cpp
)

set(
  headers
  ld52_callback.hpp
  ld52_camera_controller.hpp
  ld52_gui.hpp
  veggie_state.hpp
)

set(
  target
  ld52
)

set(LD_SCENARIO_NAME ld52_map.json)

set(LD_VIZ_CONFIG_NAME viz_config.json)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${LD_SCENARIO_NAME} ${CMAKE_BINARY_DIR}/${LD_SCENARIO_NAME} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${LD_VIZ_CONFIG_NAME} ${CMAKE_BINARY_DIR}/${LD_VIZ_CONFIG_NAME} COPYONLY)

add_executable(
  ${target}
  ${sources}
  ${headers}
)

target_compile_definitions(${target} PRIVATE SCENARIO_NAME="${LD_SCENARIO_NAME}")
target_compile_definitions(${target} PRIVATE VIZ_CONFIG_NAME="${LD_VIZ_CONFIG_NAME}")

target_link_libraries(
  ${target}
  slingshot
)
