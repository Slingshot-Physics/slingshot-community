#include "data_header.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_<type>(data_<type>_t * data)
{
   memset(data, 0, sizeof(data_<type>_t));
}

int <type>_to_json(json_value_t * node, const data_<type>_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_<type>_t")) return 0;
   // if (!add_fixed_float_array_field(node, "v", 3, data->v)) return 0;
   return 1;
}

int <type>_from_json(const json_value_t * node, data_<type>_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_<type>_t")) return 0;
   // if (!copy_fixed_float_array_field(node, "v", 3, data->v)) return 0;
   return 1;
}

int anon_<type>_to_json(json_value_t * node, const void * anon_data)
{
   const data_<type>_t * data = (const data_<type>_t *)anon_data;
   return <type>_to_json(node, data);
}

int anon_<type>_from_json(const json_value_t * node, void * anon_data)
{
   data_<type>_t * data = (data_<type>_t *)anon_data;
   return <type>_from_json(node, data);
}
