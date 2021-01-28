#include "slingshot.hpp"

#include <iostream>

int main(int argc, char ** argv)
{

   if (argc < 2)
   {
      std::cout << "command: scenario_file.json\n";
      return 0;
   }

   data_scenario_t scenario;
   int scenario_read_result = read_data_from_file(&scenario, argv[1]);
   if (!scenario_read_result)
   {
      return 1;
   }

   slingshot::api fzx(scenario, nullptr);

   fzx.loop();

   return 0;
}
