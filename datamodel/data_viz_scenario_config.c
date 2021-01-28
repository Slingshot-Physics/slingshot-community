#include "data_viz_scenario_config.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_vizScenarioConfig(data_vizScenarioConfig_t * data)
{
   memset(data, 0, sizeof(data_vizScenarioConfig_t));
}

int vizScenarioConfig_to_json(json_value_t * node, const data_vizScenarioConfig_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_vizScenarioConfig_t")) return 0;
   if (!add_object_field(node, "scenario", anon_scenario_to_json, data)) return 0;
   if (!add_object_field(node, "vizConfig", anon_vizConfig_to_json, data)) return 0;
   return 1;
}

int vizScenarioConfig_from_json(const json_value_t * node, data_vizScenarioConfig_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_vizScenarioConfig_t")) return 0;
   if (!copy_object_field(node, "scenario", anon_scenario_from_json, data)) return 0;
   if (!copy_object_field(node, "vizConfig", anon_vizConfig_from_json, data)) return 0;
   return 1;
}

int anon_vizScenarioConfig_to_json(json_value_t * node, const void * anon_data)
{
   const data_vizScenarioConfig_t * data = (const data_vizScenarioConfig_t *)anon_data;
   return vizScenarioConfig_to_json(node, data);
}

int anon_vizScenarioConfig_from_json(const json_value_t * node, void * anon_data)
{
   data_vizScenarioConfig_t * data = (data_vizScenarioConfig_t *)anon_data;
   return vizScenarioConfig_from_json(node, data);
}
