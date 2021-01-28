#include "static_logger.hpp"

#include <map>

namespace logger
{

   unsigned int defaultFrameCounter(void)
   {
      static unsigned int counter = 0;
      return counter++;
   }

   static std::map<std::string, JankLogger> _profileLoggers;

   static logger::types::loggerConfig_t _loggerConfig;

   void setLoggerConfig(const logger::types::loggerConfig_t & logger_config)
   {
      _loggerConfig = logger_config;
      if (logger_config.frameCounter == nullptr)
      {
         _loggerConfig.frameCounter = &defaultFrameCounter;
      }
   }

   void startTimer(
      int bodyIdA, int bodyIdB, const char * logger_name
   )
   {
      std::string logger_name_str(logger_name);
      if (_loggerConfig.logType == logger::types::logging::PROFILE)
      {
         unsigned int frame_count = 0;
         if (_loggerConfig.frameCounter != nullptr)
         {
            frame_count = _loggerConfig.frameCounter();
         }

         getProfileLogger(logger_name_str).StartTimer(
            bodyIdA, bodyIdB, frame_count
         );
      }
   }

   void stopTimer(const char * logger_name)
   {
      std::string logger_name_str(logger_name);
      if (_loggerConfig.logType == logger::types::logging::PROFILE)
      {
         getProfileLogger(logger_name_str).StopTimer();
      }
   }

   JankLogger & getProfileLogger(const std::string & logger_name)
   {
      if (_profileLoggers.find(logger_name) != _profileLoggers.end())
      {
         return _profileLoggers[logger_name];
      }

      std::string file_name = logger_name + "_profile.log";
      _profileLoggers[logger_name] = JankLogger(_loggerConfig.logDir, file_name, logger_name);
      return _profileLoggers[logger_name];
   }

}
