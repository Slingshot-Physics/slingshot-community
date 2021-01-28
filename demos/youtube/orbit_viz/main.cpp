#include "slingshot.hpp"
#include "simple_callback.hpp"

#include <iostream>

int main(int argc, char ** argv)
{
   if (argc < 3)
   {
      std::cout << "command: scenario_file.json viz_config_file.json\n";
      return 0;
   }

   data_scenario_t scenario;
   int scenario_read_result = read_data_from_file(&scenario, argv[1]);
   if (!scenario_read_result)
   {
      return 1;
   }

   data_vizConfig_t viz_config;
   int viz_config_read_result = read_data_from_file(&viz_config, argv[2]);
   if (!viz_config_read_result)
   {
      return 1;
   }

   std::cout << "viz config from main: " << viz_config.windowWidth << ", " << viz_config.windowHeight << "\n";

   SimpleCallback simple_cb;
   slingshot::CallbackBase * simple_cb_base = &simple_cb;

   slingshot::api fzx(scenario, viz_config, simple_cb_base);

   fzx.loop();

   return 0;
}
