#ifndef DATA_SHAPE_HEADER
#define DATA_SHAPE_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_enums.h"
#include "data_shape_capsule.h"
#include "data_shape_cube.h"
#include "data_shape_cylinder.h"
#include "data_shape_sphere.h"

typedef struct data_shape_s
{
   unsigned int bodyId;
   data_shapeType_t shapeType;

   union
   {
      data_shapeCapsule_t capsule;
      data_shapeCube_t cube;
      data_shapeCylinder_t cylinder;
      data_shapeSphere_t sphere;
   };
} data_shape_t;

void initialize_shape(data_shape_t * data);

int shape_to_json(json_value_t * node, const data_shape_t * data);

int shape_from_json(const json_value_t * node, data_shape_t * data);

int anon_shape_to_json(json_value_t * node, const void * anon_data);

int anon_shape_from_json(const json_value_t * node, void * anon_data);

#ifdef __cplusplus
}
#endif

#endif
