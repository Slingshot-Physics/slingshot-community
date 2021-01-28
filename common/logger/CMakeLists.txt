set(
   headers
   include/folders.hpp
   include/jank_logger.hpp
   include/logger.hpp
   include/logger_types.hpp
   include/logger_utils.hpp
   include/static_logger.hpp
)

set(
   sources
   src/folders.cpp
   src/jank_logger.cpp
   src/logger.cpp
   src/logger_utils.cpp
   src/static_logger.cpp
)

set(
   target
   edbdlogger
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_library(
   ${target}
   SHARED
   ${sources}
   ${headers}
)

target_include_directories(
   ${target}
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/include
)
