#include "data_logger_config.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_loggerConfig(data_loggerConfig_t * data)
{
   memset(data, 0, sizeof(data_loggerConfig_t));
}

int loggerConfig_to_json(json_value_t * node, const data_loggerConfig_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_loggerConfig_t")) return 0;
   if (!add_string_field(node, "logDir", &(data->logDir[0]))) return 0;
   if (!add_int_field(node, "loggingType", (int )data->loggingType)) return 0;
   return 1;
}

int loggerConfig_from_json(const json_value_t * node, data_loggerConfig_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_loggerConfig_t")) return 0;
   if (!copy_string_field(node, "logDir", &(data->logDir[0]), 120)) return 0;
   int temp_int;
   if (!copy_int_field(node, "loggingType", &temp_int)) return 0;
   data->loggingType = (data_loggingType_t )temp_int;
   return 1;
}

int anon_loggerConfig_to_json(json_value_t * node, const void * anon_data)
{
   const data_loggerConfig_t * data = (const data_loggerConfig_t *)anon_data;
   return loggerConfig_to_json(node, data);
}

int anon_loggerConfig_from_json(const json_value_t * node, void * anon_data)
{
   data_loggerConfig_t * data = (data_loggerConfig_t *)anon_data;
   return loggerConfig_from_json(node, data);
}
