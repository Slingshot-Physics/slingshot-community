#include "data_viz_config.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

#include <stdio.h>

void initialize_vizConfig(data_vizConfig_t * data)
{
   memset(data, 0, sizeof(data_vizConfig_t));
}

int vizConfig_to_json(json_value_t * node, const data_vizConfig_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_vizConfig_t")) return 0;
   if (!add_uint_field(node, "maxFps", data->maxFps)) return 0;
   if (!add_uint_field(node, "realtime", data->realtime)) return 0;
   if (!add_uint_field(node, "windowWidth", data->windowWidth)) return 0;
   if (!add_uint_field(node, "windowHeight", data->windowHeight)) return 0;
   if (!add_object_field(node, "cameraPos", anon_vector3_to_json, &(data->cameraPos))) return 0;
   if (!add_object_field(node, "cameraPoint", anon_vector3_to_json, &(data->cameraPoint))) return 0;
   if (!add_uint_field(node, "mousePick", data->mousePick)) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "meshProps",
         data->numMeshProps,
         5000,
         anon_vizMeshProperties_to_json,
         data->meshProps,
         sizeof(data->meshProps[0])
      )
   ) return 0;

   return 1;
}

int vizConfig_from_json(const json_value_t * node, data_vizConfig_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_vizConfig_t")) return 0;
   if (!copy_uint_field(node, "maxFps", &(data->maxFps))) return 0;
   if (!copy_uint_field(node, "realtime", &(data->realtime))) return 0;
   if (!copy_uint_field(node, "windowWidth", &(data->windowWidth))) return 0;
   if (!copy_uint_field(node, "windowHeight", &(data->windowHeight))) return 0;
   if (!copy_object_field(node, "cameraPos", anon_vector3_from_json, &(data->cameraPos))) return 0;
   if (!copy_object_field(node, "cameraPoint", anon_vector3_from_json, &(data->cameraPoint))) return 0;
   if (!copy_uint_field(node, "mousePick", &(data->mousePick))) return 0;

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "meshProps",
         5000,
         &(data->numMeshProps),
         anon_vizMeshProperties_from_json,
         (void **)&(data->meshProps),
         sizeof(data->meshProps[0])
      )
   ) return 0;

   return 1;
}

int anon_vizConfig_to_json(json_value_t * node, const void * anon_data)
{
   const data_vizConfig_t * data = (const data_vizConfig_t *)anon_data;
   return vizConfig_to_json(node, data);
}

int anon_vizConfig_from_json(const json_value_t * node, void * anon_data)
{
   data_vizConfig_t * data = (data_vizConfig_t *)anon_data;
   return vizConfig_from_json(node, data);
}

void clear_vizConfig(data_vizConfig_t * data)
{
   if (data->meshProps != NULL)
   {
      free(data->meshProps);
      data->numMeshProps = 0;
   }

   initialize_vizConfig(data);
}

void copy_vizConfig(
   const data_vizConfig_t * src, data_vizConfig_t * dest
)
{
   *dest = *src;
   if (dest->numMeshProps > 0)
   {
      dest->meshProps = (data_vizMeshProperties_t *)malloc(
         sizeof(data_vizMeshProperties_t ) * dest->numMeshProps
      );

      for (unsigned int i = 0; i < dest->numMeshProps; ++i)
      {
         dest->meshProps[i] = src->meshProps[i];
      }
   }
}
