set(
   target
   geometry_demo
)

set( source main.cpp)

add_executable(
   ${target}
   ${source}
   ${header}
)

target_link_libraries(
   ${target}
   physics
   viz
   ${lib_targets}
)
