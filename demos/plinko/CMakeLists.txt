set( target plinko )

set(
   sources
   main.cpp
   plinko_callback.cpp
)

set(
   header
   plinko_callback.hpp
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
