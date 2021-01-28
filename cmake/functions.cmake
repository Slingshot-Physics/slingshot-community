# Simple function that generates an executable given a target name, sources,
# headers, and libraries to link.
function( build_slingshot_test target_name sources headers links config_files )

   message( STATUS "binary dir for ${target_name}: ${CMAKE_BINARY_DIR}")

   set( target ${target_name} )

   add_executable(
      ${target}
      ${sources}
      ${headers}
   )

   set( test_binary_folder ${CMAKE_BINARY_DIR}/exec_tests )

   set_target_properties(
      ${target}
      PROPERTIES
      # Prompt the compiler/linker about the linker language
      LINKER_LANGUAGE CXX
      # Throw all of the executable tests into one folder
      RUNTIME_OUTPUT_DIRECTORY ${test_binary_folder}
   )
   
   target_link_libraries(
      ${target}
      ${links}
   )

   list( LENGTH config_files num_config_files )
   if ( ${num_config_files} GREATER 0 )

      string( JOIN "\",\"" config_files_str ${config_files} )

      message( "Adding ${num_config_files} files to target ${target}")
      target_compile_definitions(
         ${target}
         PRIVATE
         TEST_FILENAMES="${config_files_str}"
         PRIVATE
         NUM_FILENAMES=${num_config_files}
      )
   endif()

endfunction()
