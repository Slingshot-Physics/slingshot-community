#include "slingshot.hpp"
#include "ld52_callback.hpp"

#include <iostream>

#ifndef SCENARIO_NAME
#define SCENARIO_NAME ""
#endif

#ifndef VIZ_CONFIG_NAME
#define VIZ_CONFIG_NAME ""
#endif

int main(int argc, char ** argv)
{
   std::cout << "scenario name: " << SCENARIO_NAME << "\n";

   data_scenario_t scenario;
   data_vizConfig_t viz_config;

   if (argc == 3)
   {
      int scenario_read_result = read_data_from_file(&scenario, argv[1]);
      if (!scenario_read_result)
      {
         std::cout << "couldn't open file: " << argv[1] << "\n";
         clear_scenario(&scenario);
         return -1;
      }

      int viz_config_read_result = read_data_from_file(&viz_config, argv[2]);
      if (!viz_config_read_result)
      {
         std::cout << "couldn't open file " << argv[2] << "\n";
         clear_vizConfig(&viz_config);
         return -1;
      }
   }
   else
   {
      int scenario_read_result = read_data_from_file(&scenario, SCENARIO_NAME);
      if (!scenario_read_result)
      {
         std::cout << "couldn't open file: " << SCENARIO_NAME << "\n";
         clear_scenario(&scenario);
         return -1;
      }

      int viz_config_read_result = read_data_from_file(&viz_config, VIZ_CONFIG_NAME);
      if (!viz_config_read_result)
      {
         std::cout << "couldn't open file " << VIZ_CONFIG_NAME << "\n";
         clear_vizConfig(&viz_config);
         return -1;
      }
   }

   BuggyCallback buggy_cb;
   slingshot::CallbackBase * buggy_cb_base = &buggy_cb;

   slingshot::api fzx(scenario, viz_config, buggy_cb_base);

   fzx.loop();

   return 0;
}
