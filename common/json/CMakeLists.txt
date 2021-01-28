set(
   headers
   include/json_array.h
   include/json_char_ops.h
   include/json_deserialize.h
   include/json_object.h
   include/json_pointer.h
   include/json_serialize.h
   include/json_string.h
   include/json_types.h
   include/json_value.h
   include/json_verify.h
)

set(
   sources
   src/json_array.c
   src/json_char_ops.c
   src/json_deserialize.c
   src/json_object.c
   src/json_pointer.c
   src/json_serialize.c
   src/json_string.c
   src/json_value.c
   src/json_verify.c
)

set(
   target
   edbdjson
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

set_target_properties(
   ${target}
   PROPERTIES
   LINKER_LANGUAGE
   C
)

target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function -Werror)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/include
)
