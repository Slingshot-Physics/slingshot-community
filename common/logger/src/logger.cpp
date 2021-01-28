#include "logger.hpp"

Logger::Logger(const std::string & filename, const std::string & ownerName)
   : isParent_(true)
   , logFileRef_(logFile_)
   , decorator_(ownerName)
{
   logFile_.open(filename.c_str(), std::ofstream::out);
}

Logger::Logger(Logger & parent, const std::string & ownerName)
   : isParent_(false)
   , logFileRef_(parent.logFileRef_)
   , decorator_(parent.decorator_ + "." + ownerName)
{
}

Logger::~Logger(void)
{
   if (logFileRef_.is_open())
   {
      logFileRef_.close();
   }
}

Logger & Logger::write(void)
{
   return (*this);
}
