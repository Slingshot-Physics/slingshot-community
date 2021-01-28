#include <iostream>

#include "editor.hpp"

int main(void)
{
   data_scenario_t scenario;
   initialize_scenario(&scenario);

   data_vizConfig_t viz_config;
   initialize_vizConfig(&viz_config);
   viz_config.maxFps = 60;
   viz_config.numMeshProps = 0;
   viz_config.realtime = 1;
   viz_config.windowWidth = 800;
   viz_config.windowHeight = 600;
   viz_config.cameraPoint.v[0] = 0.f;
   viz_config.cameraPoint.v[1] = 0.f;
   viz_config.cameraPoint.v[2] = 0.f;
   viz_config.cameraPos.v[0] = 5.f;
   viz_config.cameraPos.v[1] = 5.f;
   viz_config.cameraPos.v[2] = 5.f;

   Editor editor;

   editor.loop();

   return 0;
}
