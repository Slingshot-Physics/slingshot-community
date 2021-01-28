#ifndef LOGGER_TYPES_HEADER
#define LOGGER_TYPES_HEADER

namespace logger
{

namespace types
{

   typedef unsigned int (*frameCounter_f)(void);

   namespace logging
   {
      typedef enum logging
      {
         NONE = 0,
         PROFILE = 1,
      } enum_t;
   }

   struct loggerConfig_t
   {
      loggerConfig_t(void)
         : logDir("")
         , logType(logging::enum_t::NONE)
         , frameCounter(nullptr)
      {}

      std::string logDir;
      logging::enum_t logType;
      frameCounter_f frameCounter;
   };

}

}

#endif
