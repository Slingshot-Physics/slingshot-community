set(
   target
   constraint_stove_tests
)

set(
   lib_links
   physics
)

set(
   config_files
)

build_slingshot_test( ${target} "${target}.cpp" "" "${lib_links}" "${config_files}")
