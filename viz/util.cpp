#include "util.hpp"

void readFileToString(
   const char * filePath,
   std::string & outString
)
{
   std::ifstream inFile;
   inFile.exceptions(
      std::ifstream::failbit | std::ifstream::badbit
   );
   inFile.open(filePath);
   std::stringstream fileStream;
   fileStream << inFile.rdbuf();
   inFile.close();
   outString = fileStream.str();
}
