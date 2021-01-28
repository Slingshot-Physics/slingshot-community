set(
   sources
   main.cpp
)

set(
   target
   hull_generator
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
