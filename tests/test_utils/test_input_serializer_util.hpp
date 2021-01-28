#ifndef TEST_INPUT_SERIALIZER_UTIL_HEADER
#define TEST_INPUT_SERIALIZER_UTIL_HEADER

#include "geometry_type_converters.hpp"
#include "data_model_io.h"

#include <string>

namespace test_utils
{
   template <typename InternalType_t, typename PodType_t>
   void serialize_geometry_input(
      const InternalType_t & input,
      const char * type_name,
      const std::string & filename
   )
   {
      PodType_t input_data;
      geometry::converters::to_pod(input, &input_data);
      write_data_to_file(&input_data, type_name, filename.c_str());
   }
}

#endif
