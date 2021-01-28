set(
   headers
   slingshot_type_converters.hpp
   slingshot_types.hpp
)

set(
   sources
   slingshot_type_converters.cpp
)

set(
   target
   slingshottypes
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
   ${target}
   PROPERTIES
   LINKER_LANGUAGE
   CXX
)

target_link_libraries(
   ${target}
   datamodel
   edbdgeometry
   edbdmath
   edbdlogger
   trecs
)
