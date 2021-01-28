set(
   target
   quickhull_tests
)

set(
   lib_links
   edbdgeometry
)

set(
   config_files
)

build_slingshot_test( ${target} "${target}.cpp" "" "${lib_links}" "${config_files}")
