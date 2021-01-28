#include "logger_utils.hpp"

#include <time.h>

namespace logger
{

namespace utils
{

   std::string timeDateString(void)
   {
      time_t now = time(0);
      struct tm time_struct;
      char filename_buffer[64];
      std::memset(filename_buffer, 0, 64 * sizeof(char));

      time_struct = *localtime(&now);

      int result = sprintf(
         filename_buffer,
         "%04i_%02i_%02i_%02i_%02i_%02i",
         time_struct.tm_year + 1900,
         time_struct.tm_mon + 1,
         time_struct.tm_mday,
         time_struct.tm_hour,
         time_struct.tm_min,
         time_struct.tm_sec
      );

      if (result < 0)
      {
         std::cout << "Couldn't make a time-date string.\n";
      }

      std::string filename(filename_buffer);

      return filename;
   }

   std::string timeDateFilename(
      const std::string & prefix,
      const std::string & suffix,
      const std::string & extension
   )
   {
      std::string filename(timeDateString());

      if (prefix.length() > 0)
      {
         filename.insert(0, "_");
         filename.insert(0, prefix);
      }
      if (suffix.length() > 0)
      {
         filename.append("_");
         filename.append(suffix);
      }
      if (extension.length() > 0)
      {
         filename.append(".");
         filename.append(extension);
      }

      return filename;
   }

}

}

