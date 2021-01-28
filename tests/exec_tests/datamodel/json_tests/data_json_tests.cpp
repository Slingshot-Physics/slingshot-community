#include "data_model.h"

#include "geometry_type_converters.hpp"
#include "json_deserialize.h"
#include "json_serialize.h"
#include "json_string.h"
#include "json_value.h"
#include "mesh.hpp"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <stdio.h>

TEST_CASE( "vec3 --> json" )
{
   data_vector3_t temp_vec;
   temp_vec.v[0] = -1.f;
   temp_vec.v[1] = 21.24356f;
   temp_vec.v[2] = -17.2f;

   json_value_t tree;
   json_value_initialize(&tree);
   json_value_allocate_type(&tree, JSON_OBJECT);

   vector3_to_json(&tree, &temp_vec);

   json_value_t * result;
   REQUIRE(json_value_json_pointer_access(&tree, "/_typename", &result) > 0);
   REQUIRE(result->value_type == JSON_STRING);
   json_string_t temp_str;
   json_string_initialize(&temp_str);
   json_string_allocate(&temp_str);
   json_string_make(&temp_str, "data_vector3_t");
   REQUIRE(json_string_compare(&(result->string), &temp_str) == 1);

   REQUIRE(json_value_json_pointer_access(&tree, "/v/0", &result) > 0);
   REQUIRE(result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(result->fnum - temp_vec.v[0]) < 1e-6f);

   REQUIRE(json_value_json_pointer_access(&tree, "/v/1", &result) > 0);
   REQUIRE(result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(result->fnum - temp_vec.v[1]) < 1e-6f);

   REQUIRE(json_value_json_pointer_access(&tree, "/v/2", &result) > 0);
   REQUIRE(result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(result->fnum - temp_vec.v[2]) < 1e-6f);

   json_string_t json_str;
   json_serialize_to_str(&tree, &json_str);

   printf("serialized:\n%s\n", json_str.buffer);

   json_string_delete(&temp_str);
   json_string_delete(&json_str);
   json_value_delete(&tree);
}

TEST_CASE( "json --> vec3" )
{
   json_string_t json_str;
   json_string_initialize(&json_str);
   json_string_allocate(&json_str);
   json_string_make(&json_str, "{\"v\":[-1.000000,21.243561,-17.200001],\"_typename\":\"data_vector3_t\"}");

   json_value_t tree;
   json_deserialize_str(&json_str, &tree);

   data_vector3_t temp_vec;
   vector3_from_json(&tree, &temp_vec);

   printf("struct: [%f, %f, %f]\n", temp_vec.v[0], temp_vec.v[1], temp_vec.v[2]);

   json_string_delete(&json_str);
   json_value_delete(&tree);
}

TEST_CASE( "rigidbody --> json" )
{
   data_rigidbody_t temp_body;
   temp_body.id = 5;
   temp_body.mass = 15.f;
   temp_body.linpos.v[0] = 1.234f;
   temp_body.linpos.v[1] = 5.620f;
   temp_body.linpos.v[2] = -15.4998f;
   temp_body.linvel.v[0] = 13.234f;
   temp_body.linvel.v[1] = -25.277f;
   temp_body.linvel.v[2] = -55.7928f;
   temp_body.J.m[0][0] = 1.f;
   temp_body.J.m[0][1] = 0.1f;
   temp_body.J.m[1][0] = 0.1f;
   temp_body.J.m[1][1] = 2.f;
   temp_body.J.m[2][2] = 3.f;
   temp_body.stationary = 0;
   temp_body.orientation.q[0] = 0.5f;
   temp_body.orientation.q[1] = 0.866f;
   temp_body.orientation.q[2] = 0.f;
   temp_body.orientation.q[3] = 0.f;

   json_value_t tree;
   json_value_initialize(&tree);
   json_value_allocate_type(&tree, JSON_OBJECT);

   rigidbody_to_json(&tree, &temp_body);

   json_value_t * pointer_result;
   REQUIRE(json_value_json_pointer_access(&tree, "/id", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_INT_NUMBER);
   REQUIRE(pointer_result->inum == 5);

   REQUIRE(json_value_json_pointer_access(&tree, "/linpos/v/0", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(pointer_result->fnum - temp_body.linpos.v[0]) < 1e-6f);

   REQUIRE(json_value_json_pointer_access(&tree, "/linpos/v/1", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(pointer_result->fnum - temp_body.linpos.v[1]) < 1e-6f);

   REQUIRE(json_value_json_pointer_access(&tree, "/linpos/v/2", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(pointer_result->fnum - temp_body.linpos.v[2]) < 1e-6f);

   REQUIRE(json_value_json_pointer_access(&tree, "/linvel/v/0", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(pointer_result->fnum - temp_body.linvel.v[0]) < 1e-6f);

   REQUIRE(json_value_json_pointer_access(&tree, "/linvel/v/1", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(pointer_result->fnum - temp_body.linvel.v[1]) < 1e-6f);

   REQUIRE(json_value_json_pointer_access(&tree, "/linvel/v/2", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_FLOAT_NUMBER);
   REQUIRE(fabs(pointer_result->fnum - temp_body.linvel.v[2]) < 1e-6f);

   json_string_t json_str;
   json_serialize_to_str(&tree, &json_str);

   printf("serialized:\n%s\n", json_str.buffer);

   json_string_delete(&json_str);
   json_value_delete(&tree);

}

TEST_CASE( "json --> rigidbody" )
{
   json_string_t json_str;
   json_string_initialize(&json_str);
   json_string_allocate(&json_str);
   json_string_make(
      &json_str,
      "{\"_typename\":\"data_rigidbody_t\",\"id\":5,\"J\":{\"_typename\":\"data_matrix33_t\",\"m\":[1.000000,0.100000,0.000000,0.100000,2.000000,0.000000,0.000000,0.000000,3.000000]},\"linpos\":{\"_typename\":\"data_vector3_t\",\"v\":[1.234000,5.620000,-15.499800]},\"linvel\":{\"_typename\":\"data_vector3_t\",\"v\":[13.234000,-25.277000,-55.792801]},\"linacc\":{\"_typename\":\"data_vector3_t\",\"v\":[0.000000,0.000000,0.000000]},\"orientation\":{\"_typename\":\"data_quaternion_t\",\"q\":[0.500000,0.866000,0.000000,0.000000]},\"rotvel\":{\"_typename\":\"data_vector3_t\",\"v\":[0.000000,0.000000,0.000000]},\"rotacc\":{\"_typename\":\"data_vector3_t\",\"v\":[0.000000,0.000000,0.000000]},\"stationary\":0,\"mass\":15.000000,\"gravity\":{\"_typename\":\"data_vector3_t\",\"v\":[0.000000,-0.000000,0.000000]}}"
   );

   json_value_t tree;
   json_deserialize_str(&json_str, &tree);

   data_rigidbody_t temp_body;
   rigidbody_from_json(&tree, &temp_body);

   printf("linpos: [%f, %f, %f]\n", temp_body.linpos.v[0], temp_body.linpos.v[1], temp_body.linpos.v[2]);

   json_value_delete(&tree);
   json_string_delete(&json_str);

}

TEST_CASE( "mesh --> json" )
{
   data_triangleMesh_t temp_mesh_data;
   geometry::types::triangleMesh_t temp_mesh = geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::SPHERE);
   geometry::converters::to_pod(temp_mesh, &temp_mesh_data);

   json_value_t tree;
   json_value_initialize(&tree);
   json_value_allocate_type(&tree, JSON_OBJECT);

   triangleMesh_to_json(&tree, &temp_mesh_data);

   json_value_t * pointer_result;
   REQUIRE(json_value_json_pointer_access(&tree, "/bodyId", &pointer_result) < 0);
   REQUIRE((pointer_result == nullptr));

   REQUIRE(json_value_json_pointer_access(&tree, "/numVerts", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_INT_NUMBER);
   REQUIRE(pointer_result->inum == temp_mesh.numVerts);

   REQUIRE(json_value_json_pointer_access(&tree, "/numTriangles", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_INT_NUMBER);
   REQUIRE(pointer_result->inum == temp_mesh.numTriangles);

   REQUIRE(json_value_json_pointer_access(&tree, "/verts", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_ARRAY);
   REQUIRE(pointer_result->array.size == temp_mesh.numVerts);

   REQUIRE(json_value_json_pointer_access(&tree, "/triangles", &pointer_result) > 0);
   REQUIRE(pointer_result->value_type == JSON_ARRAY);
   REQUIRE(pointer_result->array.size == temp_mesh.numTriangles);

   for (unsigned int i = 0; i < temp_mesh.numTriangles; ++i)
   {
      char temp_tri_str[32];
      snprintf(temp_tri_str, 32, "/triangles/%u", i);
      REQUIRE(json_value_json_pointer_access(&tree, temp_tri_str, &pointer_result) > 0);
      REQUIRE(pointer_result->value_type == JSON_OBJECT);

      for (unsigned int j = 0; j < 3; ++j)
      {
         char temp_tri_vertid_str[32];
         snprintf(temp_tri_vertid_str, 32, "/triangles/%u/vertIds/%u", i, j);
         REQUIRE(json_value_json_pointer_access(&tree, temp_tri_vertid_str, &pointer_result) > 0);
         REQUIRE(pointer_result->value_type == JSON_INT_NUMBER);
         REQUIRE(pointer_result->inum == temp_mesh.triangles[i].vertIds[j]);
      }

      char temp_tri_norm_str[32];
   }

   json_value_delete(&tree);

}

TEST_CASE( "json --> scenario")
{
   data_scenario_t temp_scenario;
   initialize_scenario(&temp_scenario);

   temp_scenario.numBodies = 10;
   temp_scenario.bodies = (data_rigidbody_t *)malloc(temp_scenario.numBodies * sizeof(data_rigidbody_t));

   for (unsigned int i = 0; i < temp_scenario.numBodies; ++i)
   {
      data_rigidbody_t temp_body;
      temp_body.id = 5 + i;
      temp_body.mass = 15.f;
      temp_body.linpos.v[0] = 1.234f;
      temp_body.linpos.v[1] = 5.620f;
      temp_body.linpos.v[2] = -15.4998f;
      temp_body.linvel.v[0] = 13.234f;
      temp_body.linvel.v[1] = -25.277f;
      temp_body.linvel.v[2] = -55.7928f;
      temp_body.J.m[0][0] = 1.f;
      temp_body.J.m[0][1] = 0.1f;
      temp_body.J.m[1][0] = 0.1f;
      temp_body.J.m[1][1] = 2.f;
      temp_body.J.m[2][2] = 3.f;
      temp_body.stationary = 0;
      temp_body.orientation.q[0] = 0.5f;
      temp_body.orientation.q[1] = 0.866f;
      temp_body.orientation.q[2] = 0.f;
      temp_body.orientation.q[3] = 0.f;
      temp_scenario.bodies[i] = temp_body;
   }

   json_value_t tree;
   json_value_initialize(&tree);
   json_value_allocate_type(&tree, JSON_OBJECT);
   scenario_to_json(&tree, &temp_scenario);

   json_string_t json_str;

   json_serialize_to_str(&tree, &json_str);

   printf("scenario:\n%s\n", json_str.buffer);

   json_string_delete(&json_str);
   clear_scenario(&temp_scenario);
   json_value_delete(&tree);
}

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct arbitrary
{
   float thing;
   int other_thing;
   float stuff[4];
} arbitrary_t;

int arbitrary_to_json(json_value_t * node, const arbitrary_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "arbitrary_t")) return 0;
   if (!add_float_field(node, "thing", data->thing)) return 0;
   if (!add_int_field(node, "other_thing", data->other_thing)) return 0;
   if (!add_fixed_float_array_field(node, "stuff", 4, data->stuff)) return 0;

   return 1;
}

int arbitrary_from_json(const json_value_t * node, arbitrary_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "arbitrary_t")) return 0;
   if (!copy_float_field(node, "thing", &(data->thing))) return 0;
   if (!copy_int_field(node, "other_thing", &(data->other_thing))) return 0;
   if (!copy_fixed_float_array_field(node, "stuff", 4, data->stuff)) return 0;

   return 1;
}

int anon_arbitrary_to_json(json_value_t * node, const void * anon_data)
{
   const arbitrary_t * data = (const arbitrary_t *)anon_data;
   return arbitrary_to_json(node, data);
}

int anon_arbitrary_from_json(const json_value_t * node, void * anon_data)
{
   arbitrary_t * data = (arbitrary_t *)anon_data;
   return arbitrary_from_json(node, data);
}

typedef struct garbage
{
   unsigned int numThings;
   arbitrary_t * things;
} garbage_t;

int garbage_to_json(json_value_t * node, const garbage_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "garbage_t")) return 0;

   return add_optional_dynamic_object_array_field(
      node,
      "things",
      data->numThings,
      200,
      anon_arbitrary_to_json,
      data->things,
      sizeof(data->things[0])
   );
}

int garbage_from_json(const json_value_t * node, garbage_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "garbage_t")) return 0;

   return copy_optional_dynamic_object_array_field(
      node,
      "things",
      200,
      &(data->numThings),
      anon_arbitrary_from_json,
      (void **)&(data->things),
      sizeof(data->things[0])
   );
}

int anon_garbage_to_json(json_value_t * node, const void * anon_data)
{
   const garbage_t * data = (const garbage_t *)anon_data;
   return garbage_to_json(node, data);
}

int anon_garbage_from_json(const json_value_t * node, void * anon_data)
{
   garbage_t * data = (garbage_t *)anon_data;
   return garbage_from_json(node, data);
}

#ifdef __cplusplus
}
#endif

// Verifies that a non-empty optional array can be serialized from a struct
// into JSON. Uses a JSON pointer to verify that some of the desired fields
// actually exist in the JSON string.
TEST_CASE( "serialize non-empty optional array from struct to JSON" )
{
   garbage_t temp;
   temp.numThings = 2;
   temp.things = static_cast<arbitrary_t *>(
      malloc(temp.numThings * sizeof( temp.things[0] ))
   );

   temp.things[0].other_thing = 27;
   temp.things[0].stuff[0] = 1.f;
   temp.things[0].stuff[1] = 2.f;
   temp.things[0].stuff[2] = 3.f;
   temp.things[0].stuff[3] = 4.f;
   temp.things[0].thing = -77.f;

   temp.things[1].other_thing = 52;
   temp.things[1].stuff[0] = 51.f;
   temp.things[1].stuff[1] = 52.f;
   temp.things[1].stuff[2] = 53.f;
   temp.things[1].stuff[3] = 54.f;
   temp.things[1].thing = -1377.f;

   json_value_t tree;
   json_value_initialize(&tree);
   json_value_allocate_type(&tree, JSON_OBJECT);

   anon_garbage_to_json(&tree, &temp);

   json_string_t garbage_string;
   json_string_initialize(&garbage_string);

   json_serialize_to_str(&tree, &garbage_string);

   printf("as json:\n%.*s\n", garbage_string.size, garbage_string.buffer);

   json_value_t * pointer_out = NULL;

   int pointer_result = json_value_json_pointer_access(&tree, "/things/0/other_thing", &pointer_out);

   REQUIRE( pointer_result == 1 );
   REQUIRE( pointer_out->value_type == JSON_INT_NUMBER );
   REQUIRE( pointer_out->inum == 27 );

   json_value_delete(&tree);
}

// Verifies that a struct with an empty optional array results in zero fields
// generated for the JSON string.
TEST_CASE( "serialize empty optional array from struct to JSON" )
{
   garbage_t temp;
   temp.numThings = 0;

   temp.things = NULL;

   json_value_t tree;
   json_value_initialize(&tree);
   json_value_allocate_type(&tree, JSON_OBJECT);

   anon_garbage_to_json(&tree, &temp);

   json_string_t garbage_string;
   json_string_initialize(&garbage_string);

   json_serialize_to_str(&tree, &garbage_string);

   printf("as json:\n%.*s\n", garbage_string.size, garbage_string.buffer);

   json_value_t * pointer_out = NULL;

   int pointer_result = json_value_json_pointer_access(&tree, "/things/0/other_thing", &pointer_out);

   REQUIRE( pointer_result == -1 );
   REQUIRE( pointer_out == nullptr );

   json_value_delete(&tree);
}

// Verifies that a hard-coded JSON string of a type with an optional array
// deserializes into a struct with the optional array pointer set to NULL and
// the number of 'things' set to zero.
TEST_CASE( "deserialize empty optional array from JSON to struct" )
{
   json_string_t empty_garbage_string;
   json_string_initialize(&empty_garbage_string);

   json_string_make(&empty_garbage_string, "{\"_typename\":\"garbage_t\"}");

   json_value_t tree;
   json_value_initialize(&tree);
   json_deserialize_str(&empty_garbage_string, &tree);

   garbage_t garbo;

   int read_result = anon_garbage_from_json(&tree, &garbo);

   REQUIRE( read_result );

   REQUIRE( garbo.numThings == 0 );
   REQUIRE( garbo.things == nullptr );
}

// Verifies that a hard-coded JSON string of a type with an optional array
// deserializes into a struct with the optional pointer allocated correctly
// and the number of elements set to the correct value.
TEST_CASE( "deserialize non-empty optional array from JSON to struct" )
{
   json_string_t empty_garbage_string;
   json_string_initialize(&empty_garbage_string);

   json_string_make(&empty_garbage_string, "{\"_typename\":\"garbage_t\",\"things\":[{\"_typename\":\"arbitrary_t\",\"thing\":-77.000000000000,\"other_thing\":27,\"stuff\":[1.000000000000,2.000000000000,3.000000000000,4.000000000000]},{\"_typename\":\"arbitrary_t\",\"thing\":-1377.000000000000,\"other_thing\":52,\"stuff\":[51.000000000000,52.000000000000,53.000000000000,54.000000000000]}]}");

   json_value_t tree;
   json_value_initialize(&tree);
   json_deserialize_str(&empty_garbage_string, &tree);

   garbage_t garbo;

   int read_result = anon_garbage_from_json(&tree, &garbo);

   REQUIRE( read_result );

   REQUIRE( garbo.numThings == 2 );
   REQUIRE( garbo.things != nullptr );

   REQUIRE( garbo.things[0].other_thing == 27 );
   REQUIRE( garbo.things[1].other_thing == 52 );
}
