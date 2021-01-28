#include "folders.hpp"

#include <ftw.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#endif

#include <algorithm>
#include <string>
#include <vector>

namespace folders
{

   int mkdir(const std::string & dir)
   {
#ifdef _WIN32
      int err = _mkdir(dir.c_str());
#else
      int err = ::mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
      return err;
   }

   int dir_exists(const std::string & dir)
   {
#ifdef _WIN32
      DWORD ftyp = GetFileAttributesA(dir.c_str());
      if (ftyp == INVALID_FILE_ATTRIBUTES)
      {
         return 0;
      }
      return (ftyp & FILE_ATTRIBUTE_DIRECTORY);
#else
      struct stat sb;

      return stat(dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
#endif
   }

   int mkdir_parents(const std::string & dir)
   {
      std::vector<std::string> dir_tokens;

      bool first_loop = true;
      size_t first_loc = 0;
      size_t last_loc = 0;

      std::cout << "parsing dir: " << dir << "\n";
      while (first_loc != std::string::npos)
      {
         first_loc = dir.find(
            '/', std::min<std::size_t>(first_loop ? (size_t )0 : last_loc + 1, dir.length() - 1)
         );

         if (first_loc == 0)
         {
            dir_tokens.push_back(std::string("/"));
         }
         else if (first_loc < dir.length() && first_loc > 0)
         {
            dir_tokens.push_back(dir.substr(last_loc, first_loc - last_loc));
         }
         else
         {
            dir_tokens.push_back(dir.substr(last_loc + 1, dir.length() - last_loc));
         }

         last_loc = first_loc;
         first_loop = false;
      }

      std::cout << "directory tokens\n";
      for (unsigned int i = 0; i < dir_tokens.size(); ++i)
      {
         std::cout << dir_tokens[i] << "\n";
      }

      return 1;
   }

}
