set(
   target
   slingshot
)

set(
   headers
   slingshot_callback_base.hpp
   default_callback.hpp
   slingshot.hpp
)

set(
   sources
   default_callback.cpp
   slingshot.cpp
)

if (BUILD_VIZ)
   message(STATUS "including viz methods in slingshot configuration")
   list( APPEND headers default_gui.hpp )
   list( APPEND sources default_gui.cpp slingshot_viz.cpp)
endif()

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -Werror)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}
)

set(
   libs_to_link
   datamodel
   physics
)

if (BUILD_VIZ)
   message( STATUS "linking to viz in API")
   list( APPEND libs_to_link viz )
endif()

target_link_libraries( ${target} ${libs_to_link} )
