set(
   libs
   datamodel
   edbdjson
   edbdgeometry
)

build_slingshot_test( data_file_io_tests data_file_io_tests.cpp "" "${libs}"  "" )
