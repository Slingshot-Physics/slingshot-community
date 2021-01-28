set(
   target
   rk4_increment_calculator_tests
)

set(
   lib_links
   physics
)

set(
   config_files
)

build_slingshot_test( ${target} "${target}.cpp" "" "${lib_links}" "${config_files}")
