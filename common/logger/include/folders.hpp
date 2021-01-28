#ifndef FOLDERS_HEADER
#define FOLDERS_HEADER

#include <iostream>

namespace folders
{

    int mkdir(const std::string & dir);

    int dir_exists(const std::string & dir);

    int mkdir_parents(const std::string & dir);

}

#endif
