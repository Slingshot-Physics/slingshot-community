#ifndef DATA_LOGGER_CONFIG_HEADER
#define DATA_LOGGER_CONFIG_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_enums.h"

typedef struct loggerConfig
{
   char logDir[120];

   data_loggingType_t loggingType;
} data_loggerConfig_t;

void initialize_loggerConfig(data_loggerConfig_t * data);

int loggerConfig_to_json(json_value_t * node, const data_loggerConfig_t * data);

int loggerConfig_from_json(const json_value_t * node, data_loggerConfig_t * data);

int anon_loggerConfig_to_json(json_value_t * node, const void * anon_data);

int anon_loggerConfig_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
