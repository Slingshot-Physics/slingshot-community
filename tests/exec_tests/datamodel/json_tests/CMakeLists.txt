set(
   libs
   datamodel
   edbdjson
   edbdgeometry
)

build_slingshot_test( data_json_tests data_json_tests.cpp "" "${libs}"  "" )
