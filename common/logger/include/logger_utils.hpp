#ifndef LOGGER_UTILS_HEADER
#define LOGGER_UTILS_HEADER

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace logger
{

namespace utils
{
   // Convenience function that returns a string with the formatting:
   //    yyyy_mm_dd_hh_mm_ss
   std::string timeDateString(void);

   // Convenience function that returns a string with the formatting:
   //    prefix_yyyy_mm_dd_hh_mm_ss_suffix.extension
   std::string timeDateFilename(
      const std::string & prefix,
      const std::string & suffix,
      const std::string & extension
   );

   // Please don't try to use this with stupid shit like maps or vectors or
   // pointers.
   template <typename PrefixType_t, typename SuffixType_t>
   std::string timeDateFilename(
      const PrefixType_t & prefix,
      const SuffixType_t & suffix,
      const std::string & extension
   )
   {
      std::ostringstream the_output;
      std::string time_date(timeDateString());
      the_output << prefix << "_" << time_date << "_" << suffix << "." << extension;
      return the_output.str();
   }
}

}

#endif
