#include "data_shape.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

void initialize_shape(data_shape_t * data)
{
   memset(data, 0, sizeof(data_shape_t));
}

int shape_to_json(json_value_t * node, const data_shape_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_shape_t")) return 0;
   if (!add_uint_field(node, "bodyId", data->bodyId)) return 0;

   int temp_int = (int )data->shapeType;
   if (!add_int_field(node, "shapeType", temp_int)) return 0;

   data_to_json_f jsonifier = NULL;

   switch(data->shapeType)
   {
      case DATA_SHAPE_CAPSULE:
         jsonifier = anon_shapeCapsule_to_json;
         if (!add_object_field(node, "capsule", jsonifier, &(data->capsule))) return 0;
         break;
      case DATA_SHAPE_CUBE:
         jsonifier = anon_shapeCube_to_json;
         if (!add_object_field(node, "cube", jsonifier, &(data->cube))) return 0;
         break;
      case DATA_SHAPE_CYLINDER:
         jsonifier = anon_shapeCylinder_to_json;
         if (!add_object_field(node, "cylinder", jsonifier, &(data->cylinder))) return 0;
         break;
      case DATA_SHAPE_SPHERE:
         jsonifier = anon_shapeSphere_to_json;
         if (!add_object_field(node, "sphere", jsonifier, &(data->sphere))) return 0;
         break;
      default:
         jsonifier = anon_shapeSphere_to_json;
         if (!add_object_field(node, "sphere", jsonifier, &(data->sphere))) return 0;
         break;
   }

   return 1;
}

int shape_from_json(const json_value_t * node, data_shape_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_shape_t")) return 0;
   if (!copy_uint_field(node, "bodyId", &(data->bodyId))) return 0;
   int temp_int;
   if (!copy_int_field(node, "shapeType", &temp_int)) return 0;
   data->shapeType = (data_shapeType_t )temp_int;

   data_from_json_f structifier = NULL;

   switch(data->shapeType)
   {
      case DATA_SHAPE_CAPSULE:
         structifier = anon_shapeCapsule_from_json;
         if (!copy_object_field(node, "capsule", structifier, &(data->capsule))) return 0;
         break;
      case DATA_SHAPE_CUBE:
         structifier = anon_shapeCube_from_json;
         if (!copy_object_field(node, "cube", structifier, &(data->cube))) return 0;
         break;
      case DATA_SHAPE_CYLINDER:
         structifier = anon_shapeCylinder_from_json;
         if (!copy_object_field(node, "cylinder", structifier, &(data->cylinder))) return 0;
         break;
      case DATA_SHAPE_SPHERE:
         structifier = anon_shapeSphere_from_json;
         if (!copy_object_field(node, "sphere", structifier, &(data->sphere))) return 0;
         break;
      default:
         structifier = anon_shapeSphere_from_json;
         if (!copy_object_field(node, "sphere", structifier, &(data->sphere))) return 0;
         break;
   }

   return 1;
}

int anon_shape_to_json(json_value_t * node, const void * anon_data)
{
   const data_shape_t * data = (const data_shape_t *)anon_data;
   return shape_to_json(node, data);
}

int anon_shape_from_json(const json_value_t * node, void * anon_data)
{
   data_shape_t * data = (data_shape_t *)anon_data;
   return shape_from_json(node, data);
}
