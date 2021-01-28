#include "data_scenario.h"

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

#include <stdio.h>

void initialize_scenario(data_scenario_t * scenario)
{
   memset(scenario, 0, sizeof(data_scenario_t));
}

int scenario_to_json(json_value_t * node, const data_scenario_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!add_typename(node, "data_scenario_t")) return 0;
   if (!add_object_field(node, "logger", anon_loggerConfig_to_json, &(data->logger))) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "isometricColliders",
         data->numIsometricColliders,
         5000,
         anon_isometricCollider_to_json,
         data->isometricColliders,
         sizeof(data->isometricColliders[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "bodies",
         data->numBodies,
         5000,
         anon_rigidbody_to_json,
         data->bodies,
         sizeof(data->bodies[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "shapes",
         data->numShapes,
         5000,
         anon_shape_to_json,
         data->shapes,
         sizeof(data->shapes[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "balljointConstraints",
         data->numBalljointConstraints,
         5000,
         anon_constraintBalljoint_to_json,
         data->balljointConstraints,
         sizeof(data->balljointConstraints[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "gearConstraints",
         data->numGearConstraints,
         5000,
         anon_constraintGear_to_json,
         data->gearConstraints,
         sizeof(data->gearConstraints[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "revoluteJointConstraints",
         data->numRevoluteJointConstraints,
         5000,
         anon_constraintRevoluteJoint_to_json,
         data->revoluteJointConstraints,
         sizeof(data->revoluteJointConstraints[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "revoluteMotorConstraints",
         data->numRevoluteMotorConstraints,
         5000,
         anon_constraintRevoluteMotor_to_json,
         data->revoluteMotorConstraints,
         sizeof(data->revoluteMotorConstraints[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "rotation1dConstraints",
         data->numRotation1dConstraints,
         5000,
         anon_constraintRotation1d_to_json,
         data->rotation1dConstraints,
         sizeof(data->rotation1dConstraints[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "translation1dConstraints",
         data->numTranslation1dConstraints,
         5000,
         anon_constraintTranslation1d_to_json,
         data->translation1dConstraints,
         sizeof(data->translation1dConstraints[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "constantForces",
         data->numConstantForces,
         5000,
         anon_forceConstant_to_json,
         data->constantForces,
         sizeof(data->constantForces[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "dragForces",
         data->numDragForces,
         5000,
         anon_forceDrag_to_json,
         data->dragForces,
         sizeof(data->dragForces[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "springForces",
         data->numSpringForces,
         5000,
         anon_forceSpring_to_json,
         data->springForces,
         sizeof(data->springForces[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "velocityDamperForces",
         data->numVelocityDamperForces,
         5000,
         anon_forceVelocityDamper_to_json,
         data->velocityDamperForces,
         sizeof(data->velocityDamperForces[0])
      )
   ) return 0;

   if (
      !add_optional_dynamic_object_array_field(
         node,
         "dragTorques",
         data->numDragTorques,
         5000,
         anon_torqueDrag_to_json,
         data->dragTorques,
         sizeof(data->dragTorques[0])
      )
   ) return 0;

   return 1;
}

int scenario_from_json(const json_value_t * node, data_scenario_t * data)
{
   if (node == NULL) return 0;
   if (data == NULL) return 0;
   if (!verify_typename(node, "data_scenario_t")) return 0;
   if (!copy_object_field(node, "logger", anon_loggerConfig_from_json, &(data->logger))) return 0;

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "isometricColliders",
         5000,
         &(data->numIsometricColliders),
         anon_isometricCollider_from_json,
         (void **)&(data->isometricColliders),
         sizeof(data->isometricColliders[0])
      )
   )
   {
      printf("isometric colliders failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "bodies",
         5000,
         &(data->numBodies),
         anon_rigidbody_from_json,
         (void **)&(data->bodies),
         sizeof(data->bodies[0])
      )
   )
   {
      printf("rigid bodies failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "shapes",
         5000,
         &(data->numShapes),
         anon_shape_from_json,
         (void **)&(data->shapes),
         sizeof(data->shapes[0])
      )
   )
   {
      printf("shapes failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "balljointConstraints",
         5000,
         &(data->numBalljointConstraints),
         anon_constraintBalljoint_from_json,
         (void **)&(data->balljointConstraints),
         sizeof(data->balljointConstraints[0])
      )
   )
   {
      printf("ball joints failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "gearConstraints",
         5000,
         &(data->numGearConstraints),
         anon_constraintGear_from_json,
         (void **)&(data->gearConstraints),
         sizeof(data->gearConstraints[0])
      )
   )
   {
      printf("gears failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "revoluteJointConstraints",
         5000,
         &(data->numRevoluteJointConstraints),
         anon_constraintRevoluteJoint_from_json,
         (void **)&(data->revoluteJointConstraints),
         sizeof(data->revoluteJointConstraints[0])
      )
   )
   {
      printf("revolute joints failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "revoluteMotorConstraints",
         5000,
         &(data->numRevoluteMotorConstraints),
         anon_constraintRevoluteMotor_from_json,
         (void **)&(data->revoluteMotorConstraints),
         sizeof(data->revoluteMotorConstraints[0])
      )
   )
   {
      printf("revolute motors failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "rotation1dConstraints",
         5000,
         &(data->numRotation1dConstraints),
         anon_constraintRotation1d_from_json,
         (void **)&(data->rotation1dConstraints),
         sizeof(data->rotation1dConstraints[0])
      )
   )
   {
      printf("rotation 1ds failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "translation1dConstraints",
         5000,
         &(data->numTranslation1dConstraints),
         anon_constraintTranslation1d_from_json,
         (void **)&(data->translation1dConstraints),
         sizeof(data->translation1dConstraints[0])
      )
   )
   {
      printf("translation 1ds failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "constantForces",
         5000,
         &(data->numConstantForces),
         anon_forceConstant_from_json,
         (void **)&(data->constantForces),
         sizeof(data->constantForces[0])
      )
   )
   {
      printf("constant forces failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "dragForces",
         5000,
         &(data->numDragForces),
         anon_forceDrag_from_json,
         (void **)&(data->dragForces),
         sizeof(data->dragForces[0])
      )
   )
   {
      printf("drag forces failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "springForces",
         5000,
         &(data->numSpringForces),
         anon_forceSpring_from_json,
         (void **)&(data->springForces),
         sizeof(data->springForces[0])
      )
   )
   {
      printf("spring forces failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "velocityDamperForces",
         5000,
         &(data->numVelocityDamperForces),
         anon_forceVelocityDamper_from_json,
         (void **)&(data->velocityDamperForces),
         sizeof(data->velocityDamperForces[0])
      )
   )
   {
      printf("velocity dampers failed\n");
      return 0;
   }

   if (
      !copy_optional_dynamic_object_array_field(
         node,
         "dragTorques",
         5000,
         &(data->numDragTorques),
         anon_torqueDrag_from_json,
         (void **)&(data->dragTorques),
         sizeof(data->dragTorques[0])
      )
   )
   {
      printf("drag torques failed\n");
      return 0;
   }

   return 1;
}

int anon_scenario_to_json(json_value_t * node, const void * anon_data)
{
   const data_scenario_t * data = (const data_scenario_t *)anon_data;
   return scenario_to_json(node, data);
}

int anon_scenario_from_json(const json_value_t * node, void * anon_data)
{
   data_scenario_t * data = (data_scenario_t *)anon_data;
   return scenario_from_json(node, data);
}

void clear_scenario(data_scenario_t * scenario)
{
   if (scenario->isometricColliders != NULL)
   {
      free(scenario->isometricColliders);
   }
   if (scenario->bodies != NULL)
   {
      free(scenario->bodies);
   }
   if (scenario->shapes != NULL)
   {
      free(scenario->shapes);
   }
   if (scenario->balljointConstraints != NULL)
   {
      free(scenario->balljointConstraints);
   }
   if (scenario->gearConstraints != NULL)
   {
      free(scenario->gearConstraints);
   }
   if (scenario->revoluteJointConstraints != NULL)
   {
      free(scenario->revoluteJointConstraints);
   }
   if (scenario->revoluteMotorConstraints != NULL)
   {
      free(scenario->revoluteMotorConstraints);
   }
   if (scenario->springForces != NULL)
   {
      free(scenario->springForces);
   }

   initialize_scenario(scenario);
}
