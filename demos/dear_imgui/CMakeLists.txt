set(
   sources
   main.cpp
)

set(
   target
   imgui_main
)

add_executable(
   ${target}
   ${sources}
)

target_link_libraries(
   ${target}
   viz
)
