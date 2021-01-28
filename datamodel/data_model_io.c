#ifdef __cplusplus
extern "C"
{
#endif

#include "data_model_io.h"

#include "json_value.h"
#include "json_deserialize.h"
#include "json_serialize.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_TYPENAMES 44

#define MAX_TYPENAME_SIZE 120

#define TO_STR(val) #val

#define ADD_CONVERTER( type_name ) { \
   .tname=TO_STR(data_ ## type_name ## _t),\
   .jsonifier=anon_ ## type_name ## _to_json, \
   .structifier=anon_ ## type_name ## _from_json \
}

typedef struct
{
   const char tname[MAX_TYPENAME_SIZE];
   data_to_json_f jsonifier;
   data_from_json_f structifier;
} type_name_converter_t;

typedef struct
{
   unsigned int numTypenames;
   type_name_converter_t converters[NUM_TYPENAMES];
} type_names_to_converters_t;

static const type_names_to_converters_t converters = {
   .numTypenames=NUM_TYPENAMES,
   .converters={
      ADD_CONVERTER(constraintBalljoint),
      ADD_CONVERTER(constraintCollision),
      ADD_CONVERTER(constraintFriction),
      ADD_CONVERTER(constraintGear),
      ADD_CONVERTER(constraintRevoluteJoint),
      ADD_CONVERTER(constraintRevoluteMotor),
      ADD_CONVERTER(constraintRotation1d),
      ADD_CONVERTER(constraintTranslation1d),
      ADD_CONVERTER(forceConstant),
      ADD_CONVERTER(forceDrag),
      ADD_CONVERTER(forceSpring),
      ADD_CONVERTER(forceVelocityDamper),
      ADD_CONVERTER(gaussMapFace),
      ADD_CONVERTER(gaussMapMesh),
      ADD_CONVERTER(isometricCollider),
      ADD_CONVERTER(isometricTransform),
      ADD_CONVERTER(leanNeighborTriangle),
      ADD_CONVERTER(loggerConfig),
      ADD_CONVERTER(matrix33),
      ADD_CONVERTER(meshTriangle),
      ADD_CONVERTER(multiCollider),
      ADD_CONVERTER(polygon50),
      ADD_CONVERTER(quaternion),
      ADD_CONVERTER(rigidbody),
      ADD_CONVERTER(scenario),
      ADD_CONVERTER(shape),
      ADD_CONVERTER(shapeCapsule),
      ADD_CONVERTER(shapeCube),
      ADD_CONVERTER(shapeCylinder),
      ADD_CONVERTER(shapeNamed),
      ADD_CONVERTER(shapeSphere),
      ADD_CONVERTER(testGjkInput),
      ADD_CONVERTER(testTetrahedronInput),
      ADD_CONVERTER(testTriangleInput),
      ADD_CONVERTER(tetrahedron),
      ADD_CONVERTER(torqueDrag),
      ADD_CONVERTER(transform),
      ADD_CONVERTER(triangle),
      ADD_CONVERTER(triangleMesh),
      ADD_CONVERTER(vector3),
      ADD_CONVERTER(vector4),
      ADD_CONVERTER(vizConfig),
      ADD_CONVERTER(vizScenarioConfig),
      ADD_CONVERTER(vizMeshProperties),
   }
};

data_to_json_f get_jsonifier(const char * type_name)
{
   data_to_json_f jsonifier = NULL;

   for (unsigned int i = 0; i < converters.numTypenames; ++i)
   {
      if (strncmp(type_name, converters.converters[i].tname, 120) == 0)
      {
         return converters.converters[i].jsonifier;
      }
   }

   return jsonifier;
}

data_from_json_f get_structifier(const json_string_t * type_name)
{
   data_from_json_f structifier = NULL;

   for (unsigned int i = 0; i < converters.numTypenames; ++i)
   {
      if (strncmp(type_name->buffer, converters.converters[i].tname, 120) == 0)
      {
         return converters.converters[i].structifier;
      }
   }

   return structifier;
}

int write_data_to_file(
   const void * data, const char * type_name, const char * filename
)
{
   FILE * fp = fopen(filename, "w");

   if (data == NULL)
   {
      printf("Couldn't open file %s for writing\n", filename);
      fclose(fp);
      return 0;
   }

   data_to_json_f jsonifier = get_jsonifier(type_name);

   if (jsonifier == NULL)
   {
      printf("Couldn't find a jsonifier for type name %s\n", type_name);
      fclose(fp);
      return 0;
   }

   json_value_t tree;
   json_value_initialize(&tree);
   json_value_allocate_type(&tree, JSON_OBJECT);

   int jsonification_result = jsonifier(&tree, data);

   if (!jsonification_result)
   {
      printf("Couldn't jsonify the provided data\n");
      json_value_delete(&tree);
      fclose(fp);
      return 0;
   }

   json_serialize_to_file(&tree, fp);

   json_value_delete(&tree);
   fclose(fp);

   return 1;
}

int read_data_from_file(void * data, const char * filename)
{
   FILE * fp = fopen(filename, "r");

   if (fp == NULL)
   {
      printf("Couldn't open filename %s\n", filename);
      return 0;
   }

   json_value_t tree;
   json_value_initialize(&tree);
   // Don't need to allocate tree because the deserializer will do that for us.

   json_deserialize_file(fp, &tree);

   const json_value_t * type_name_value = get_const_field_by_name(&tree, "_typename");

   if (type_name_value == NULL)
   {
      printf("Couldn't find field in struct with field name \"_typename\"\n");
      fclose(fp);
      json_value_delete(&tree);
      return 0;
   }

   data_from_json_f structifier = get_structifier(&(type_name_value->string));

   if (structifier == NULL)
   {
      printf("Couldn't find a json-to-struct converter for type name: %*s\n", type_name_value->string.size, type_name_value->string.buffer);
      fclose(fp);
      json_value_delete(&tree);
      return 0;
   }

   int structification_result = structifier(&tree, data);

   if (!structification_result)
   {
      printf("Couldn't convert JSON tree to a struct\n");
      fclose(fp);
      json_value_delete(&tree);
      return 0;
   }

   json_value_delete(&tree);
   fclose(fp);

   return 1;
}

#ifdef __cplusplus
}
#endif
