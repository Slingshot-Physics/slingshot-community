#ifndef DATA_SCENARIO_HEADER
#define DATA_SCENARIO_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "json_types.h"

#include "data_constraint_balljoint.h"
#include "data_constraint_gear.h"
#include "data_constraint_revolute_joint.h"
#include "data_constraint_revolute_motor.h"
#include "data_constraint_rotation_1d.h"
#include "data_constraint_translation_1d.h"
#include "data_force_constant.h"
#include "data_force_drag.h"
#include "data_force_spring.h"
#include "data_force_velocity_damper.h"
#include "data_isometric_collider.h"
#include "data_logger_config.h"
#include "data_rigidbody.h"
#include "data_shape.h"
#include "data_torque_drag.h"

typedef struct scenario
{
   unsigned int numIsometricColliders;
   data_isometricCollider_t * isometricColliders;

   unsigned int numBodies;
   data_rigidbody_t * bodies;

   unsigned int numShapes;
   data_shape_t * shapes;

   unsigned int numBalljointConstraints;
   data_constraintBalljoint_t * balljointConstraints;

   unsigned int numGearConstraints;
   data_constraintGear_t * gearConstraints;

   unsigned int numRevoluteJointConstraints;
   data_constraintRevoluteJoint_t * revoluteJointConstraints;

   unsigned int numRevoluteMotorConstraints;
   data_constraintRevoluteMotor_t * revoluteMotorConstraints;

   unsigned int numRotation1dConstraints;
   data_constraintRotation1d_t * rotation1dConstraints;

   unsigned int numTranslation1dConstraints;
   data_constraintTranslation1d_t * translation1dConstraints;

   unsigned int numConstantForces;
   data_forceConstant_t * constantForces;

   unsigned int numDragForces;
   data_forceDrag_t * dragForces;

   unsigned int numSpringForces;
   data_forceSpring_t * springForces;

   unsigned int numVelocityDamperForces;
   data_forceVelocityDamper_t * velocityDamperForces;

   unsigned int numDragTorques;
   data_torqueDrag_t * dragTorques;

   data_loggerConfig_t logger;
} data_scenario_t;

void initialize_scenario(data_scenario_t * scenario);

int scenario_to_json(json_value_t * node, const data_scenario_t * data);

int scenario_from_json(const json_value_t * node, data_scenario_t * data);

int anon_scenario_to_json(json_value_t * node, const void * anon_data);

int anon_scenario_from_json(const json_value_t * node, void * anon_data);

void clear_scenario(data_scenario_t * scenario);

#ifdef __cplusplus
}
#endif

#endif
