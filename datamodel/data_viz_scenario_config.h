#ifndef DATA_VIZ_SCENARIO_CONFIG_HEADER
#define DATA_VIZ_SCENARIO_CONFIG_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_scenario.h"
#include "data_viz_config.h"

typedef struct data_vizScenarioConfig_s
{
   data_scenario_t scenario;
   data_vizConfig_t vizConfig;
} data_vizScenarioConfig_t;

void initialize_vizScenarioConfig(data_vizScenarioConfig_t * data);

int vizScenarioConfig_to_json(json_value_t * node, const data_vizScenarioConfig_t * data);

int vizScenarioConfig_from_json(const json_value_t * node, data_vizScenarioConfig_t * data);

int anon_vizScenarioConfig_to_json(json_value_t * node, const void * anon_data);

int anon_vizScenarioConfig_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
