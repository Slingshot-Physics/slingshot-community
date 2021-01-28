set(
   sources
   main.cpp
)

set(
   target
   mesh_main
)

add_executable(
  ${target}
  ${sources}
)

target_link_libraries(
  ${target}
  physics
  viz
)
