set(
   target
   plane_tests
)

set(
   lib_links
   edbdgeometry
   test_utils
)

set(
   config_files
)

build_slingshot_test( ${target} "${target}.cpp" "" "${lib_links}" "${config_files}")
