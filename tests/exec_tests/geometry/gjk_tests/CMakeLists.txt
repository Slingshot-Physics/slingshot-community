set(
   target
   gjk_tests
)

set(
   lib_links
   edbdgeometry
   test_utils
)

file( GLOB config_files ${CMAKE_CURRENT_SOURCE_DIR}/input_files/*.json )

build_slingshot_test( ${target} "${target}.cpp" "" "${lib_links}" "${config_files}")
