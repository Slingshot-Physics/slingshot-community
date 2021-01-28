set(
   sources
   basic_shader.cpp
   callbacks.cpp
   default_camera_controller.cpp
   flat_mesh_buffer.cpp
   glad/src/glad.c
   imgui/backends/imgui_impl_glfw.cpp
   imgui/backends/imgui_impl_opengl3.cpp
   imgui/imgui_demo.cpp
   imgui/imgui_draw.cpp
   imgui/imgui_tables.cpp
   imgui/imgui_widgets.cpp
   imgui/imgui.cpp
   imgui/misc/cpp/imgui_stdlib.cpp
   opengl_renderer.cpp
   util.cpp
   viz_renderer.cpp
   viztypeconverters.cpp
)

set(
   headers
   basic_shader.hpp
   callbacks.hpp
   camera.hpp
   default_camera_controller.hpp
   flat_mesh_buffer.hpp
   gl_common.hpp
   grid.hpp
   gui_callback_base.hpp
   hid_interface.hpp
   imgui/imgui.h
   imgui/backends/imgui_impl_glfw.h
   imgui/backends/imgui_impl_opengl3.h
   imgui/backends/imgui_impl_opengl3_loader.h
   imgui/imconfig.h
   imgui/imgui_internal.h
   imgui/imstb_rectpack.h
   imgui/imstb_textedit.h
   imgui/imstb_truetype.h
   imgui/misc/cpp/imgui_stdlib.h
   opengl_renderer.hpp
   util.hpp
   viz_renderer.hpp
   viz_types.hpp
   viztypeconverters.hpp
)

set(
   target
   viz
)

set(
   shader_names
   vertex_mvp.vs
   fragment_emitter.fs
   vertex_instance.vs
   fragment_emitter.fs
   shadow_map_fragment.fs
   shadow_map_vertex.vs
   simple_depth_shader.fs
   simple_depth_shader.vs
)

foreach( shader_name ${shader_names})
   configure_file("${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader_name}" "${CMAKE_BINARY_DIR}/${shader_name}" COPYONLY)
endforeach()

set(
   shader_dir
   "."
)

add_compile_definitions(
   SHADER_DIR=${shader_dir}
   VERTEX_MVP_SHADER_LOC="${shader_dir}/vertex_mvp.vs"
   LINE_FRAGMENT_SHADER_LOC="${shader_dir}/fragment_emitter.fs"
   INSTANCED_VERTEX_SHADER_LOC="${shader_dir}/vertex_instance.vs"
   EMITTER_FRAGMENT_SHADER_LOC="${shader_dir}/fragment_emitter.fs"
   DEPTH_FRAGMENT_SHADER_LOC="${shader_dir}/simple_depth_shader.fs"
   DEPTH_VERTEX_SHADER_LOC="${shader_dir}/simple_depth_shader.vs"
   SHADOW_MAP_FRAGMENT_SHADER_LOC="${shader_dir}/shadow_map_fragment.fs"
   SHADOW_MAP_VERTEX_SHADER_LOC="${shader_dir}/shadow_map_vertex.vs"
)

find_package(OpenGL REQUIRED)

add_subdirectory(glfw)

if (APPLE OR UNIX)
   include_directories(
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/glad/include/
      ${OpenGL_INCLUDE_DIRS}
   )
elseif (WIN32)
   include_directories(
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/glad/include/
      ${OpenGL_INCLUDE_DIRS}
   )
   message(STATUS ${OPENGL_INCLUDE_DIRS})
endif()

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
   ${CMAKE_CURRENT_SOURCE_DIR}/glad/include/
   ${CMAKE_CURRENT_SOURCE_DIR}/imgui
   ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include
   ${CMAKE_CURRENT_SOURCE_DIR}/glm
)

set_target_properties(
   ${target}
   PROPERTIES
   LINKER_LANGUAGE
   CXX
)

if (APPLE)
   set(
      uncommon_libs
      ${uncommon_libs}
      OpenGL::GL
      pthread
   )
elseif (UNIX)
   set(
      uncommon_libs
      ${uncommon_libs}
      GL
      pthread
      X11
   )
elseif (WIN32)
   set(
      uncommon_libs
      ${uncommon_libs}
      OpenGL::GL
   )
endif()

target_link_libraries(
   ${target}
   datamodel
   edbdmath
   glfw
   ${uncommon_libs}
)
