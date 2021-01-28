set(
   targets
   no_viz
)

if (BUILD_VIZ)
   list(APPEND targets viz)
endif()

foreach( target ${targets} )

   set( actual_target "api_${target}")

   set(
      sources
      ${target}.cpp
   )

   add_executable(
      ${actual_target}
      ${sources}
   )

   target_link_libraries(
      ${actual_target}
      slingshot
   )

endforeach()
