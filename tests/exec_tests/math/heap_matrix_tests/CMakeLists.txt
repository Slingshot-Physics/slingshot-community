set(
   target
   heap_matrix_tests
)

set(
   lib_links
   edbdmath
)

set(
   config_files
)

build_slingshot_test( ${target} "${target}.cpp" "" "${lib_links}" "${config_files}")
