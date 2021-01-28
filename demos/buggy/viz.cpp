#include "slingshot.hpp"
#include "buggy_callback.hpp"

#include <iostream>

int main(int argc, char ** argv)
{
   if (argc < 3)
   {
      std::cout << "command: /path/to/buggy_scenario.json /path/to/viz_config.json\n";
      return 0;
   }

   data_scenario_t scenario;
   int scenario_read_result = read_data_from_file(&scenario, argv[1]);
   if (!scenario_read_result)
   {
      std::cout << "couldn't open file: " << argv[1] << "\n";
      clear_scenario(&scenario);
      return -1;
   }

   data_vizConfig_t viz_config;
   int viz_config_read_result = read_data_from_file(&viz_config, argv[2]);
   if (!viz_config_read_result)
   {
      std::cout << "couldn't open file " << argv[2] << "\n";
      clear_vizConfig(&viz_config);
      return -1;
   }

   std::cout << "viz config from main: " << viz_config.windowWidth << ", " << viz_config.windowHeight << "\n";

   BuggyCallback buggy_cb;
   slingshot::CallbackBase * buggy_cb_base = &buggy_cb;

   slingshot::api fzx(scenario, viz_config, buggy_cb_base);

   fzx.loop();

   return 0;
}
