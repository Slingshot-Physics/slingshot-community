set(
  sources
  main.cpp
  scuffed_callback.cpp
  scuffed_camera_controller.cpp
  scuffed_gui.cpp
)

set(
  headers
  scuffed_callback.hpp
  scuffed_camera_controller.hpp
  scuffed_gui.hpp
)

set(
  target
  youtube_vid01
)

set(LD_SCENARIO_NAME scuffed_it_league.json)

set(LD_VIZ_CONFIG_NAME viz_scuffed_it_league.json)

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
