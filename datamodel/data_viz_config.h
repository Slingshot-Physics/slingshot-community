#ifndef DATA_VIZCONFIG_HEADER
#define DATA_VIZCONFIG_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_vector3.h"
#include "data_viz_mesh_properties.h"

typedef struct vizConfig
{
   unsigned int maxFps;

   unsigned int realtime;

   unsigned int windowWidth;

   unsigned int windowHeight;

   // Camera position in world coordinates (ENU).
   data_vector3_t cameraPos;

   // Point the camera is looking at (ENU).
   data_vector3_t cameraPoint;

   // Enables or disables mouse picking.
   unsigned int mousePick;

   unsigned int numMeshProps;

   data_vizMeshProperties_t * meshProps;

} data_vizConfig_t;

void initialize_vizConfig(data_vizConfig_t * data);

int vizConfig_to_json(json_value_t * node, const data_vizConfig_t * data);

int vizConfig_from_json(const json_value_t * node, data_vizConfig_t * data);

int anon_vizConfig_to_json(json_value_t * node, const void * anon_data);

int anon_vizConfig_from_json(const json_value_t * node, void * anon_data);

void clear_vizConfig(data_vizConfig_t * data);

void copy_vizConfig(
   const data_vizConfig_t * src, data_vizConfig_t * dest
);

#ifdef __cplusplus
}
#endif

#endif
