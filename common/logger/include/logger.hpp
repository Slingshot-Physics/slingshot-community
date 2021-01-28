#ifndef LOGGER
#define LOGGER

#include <iostream>
#include <fstream>

class Logger
{
   public:
      Logger(const std::string & filename, const std::string & ownerName);

      Logger(Logger & parent, const std::string & ownerName);

      ~Logger(void);

      Logger & write(void);

      template <typename T>
      Logger & operator<<(const T & v)
      {
         logFileRef_ << v;
         return *this;
      }

      Logger & operator<<(std::ostream&(*f)(std::ostream &))
      {
         logFileRef_ << f;
         return *this;
      }

   private:
      Logger(void);

      Logger(const Logger &);

      const bool isParent_;

      std::ofstream logFile_;

      std::ofstream & logFileRef_;

      std::string decorator_;

};

#endif
