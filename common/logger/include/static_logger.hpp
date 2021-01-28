#ifndef LOGGER_HEADER
#define LOGGER_HEADER

#include "jank_logger.hpp"

#include "logger_types.hpp"

#include <iostream>

namespace logger
{
   void setLoggerConfig(const logger::types::loggerConfig_t & logger_config);

   void startTimer(
      int bodyIdA, int bodyIdB, const char * logger_name
   );

   void stopTimer(const char * logger_name);

   JankLogger & getProfileLogger(const std::string & logger_name);

}

#endif
