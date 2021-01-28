#include "constraint_output.hpp"

#include "collision.hpp"
#include "friction.hpp"
#include "gear.hpp"
#include "revolute_motor.hpp"
#include "rotation_1d.hpp"
#include "torsional_friction.hpp"
#include "translation_1d.hpp"

namespace oy
{
   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintCollision_t & constraint,
      float dt
   )
   {
      oy::types::constraintOutput_t output;
      output.jacobian = oy::constraints::collision::calculateMatrixJacobian(
         parent_stationary,
         child_stationary,
         parent_id,
         child_id,
         parent_body,
         child_body,
         constraint
      );

      output.baumgarte = oy::constraints::collision::evaluateBaumgarte(
         parent_id, child_id, parent_body, child_body, constraint, dt
      );

      oy::constraints::collision::calculateLambdaBounds(
         parent_id, child_id, parent_body, child_body, output.lambdaMin, output.lambdaMax
      );

      output.constraintType = oy::types::enumConstraint_t::COLLISION;

      return output;
   }

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintFriction_t & constraint,
      float dt
   )
   {
      oy::types::constraintOutput_t output;
      output.jacobian = oy::constraints::friction::calculateMatrixJacobian(
         parent_stationary,
         child_stationary,
         parent_id,
         child_id,
         parent_body,
         child_body,
         constraint
      );

      output.baumgarte = oy::constraints::friction::evaluateBaumgarte(
         parent_id, child_id, parent_body, child_body, constraint, dt
      );

      oy::constraints::friction::calculateLambdaBounds(
         parent_id, child_id, parent_body, child_body, constraint, output.lambdaMin, output.lambdaMax
      );

      output.constraintType = oy::types::enumConstraint_t::FRICTION;

      return output;
   }

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintGear_t & constraint,
      float dt
   )
   {
      oy::types::constraintOutput_t output;
      output.jacobian = oy::constraints::gear::calculateMatrixJacobian(
         parent_stationary,
         child_stationary,
         parent_id,
         child_id,
         parent_body,
         child_body,
         constraint
      );

      output.baumgarte = oy::constraints::gear::evaluateBaumgarte(
         parent_id, child_id, parent_body, child_body, constraint, dt
      );

      oy::constraints::gear::calculateLambdaBounds(
         parent_id, child_id, parent_body, child_body, constraint, output.lambdaMin, output.lambdaMax
      );

      output.constraintType = oy::types::enumConstraint_t::GEAR;

      return output;
   }

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRevoluteMotor_t & constraint,
      float dt
   )
   {
      oy::types::constraintOutput_t output;
      output.jacobian = oy::constraints::revolute_motor::calculateMatrixJacobian(
         parent_stationary,
         child_stationary,
         parent_id,
         child_id,
         parent_body,
         child_body,
         constraint
      );

      output.baumgarte = oy::constraints::revolute_motor::evaluateBaumgarte(
         parent_id, child_id, parent_body, child_body, constraint, dt
      );

      oy::constraints::revolute_motor::calculateLambdaBounds(
         parent_id, child_id, parent_body, child_body, constraint, dt, output.lambdaMin, output.lambdaMax
      );

      output.constraintType = oy::types::enumConstraint_t::REVOLUTEMOTOR;

      return output;
   }

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintRotation1d_t & constraint,
      float dt
   )
   {
      oy::types::constraintOutput_t output;

      oy::types::rigidBody_t world;
      world.ql2b[0] = 1.f;

      oy::types::rigidBody_t temp_parent = (
         (parent_id != oy::types::null_body_entity)
         ? parent_body
         : world
      );

      oy::types::rigidBody_t temp_child = (
         (child_id != oy::types::null_body_entity)
         ? child_body
         : world
      );

      output.jacobian = oy::constraints::rotation_1d::calculateMatrixJacobian(
         parent_stationary,
         child_stationary,
         parent_id,
         child_id,
         temp_parent,
         temp_child,
         constraint
      );

      output.baumgarte = oy::constraints::rotation_1d::evaluateBaumgarte(
         parent_id, child_id, temp_parent, temp_child, constraint, dt
      );

      oy::constraints::rotation_1d::calculateLambdaBounds(
         parent_id, child_id, temp_parent, temp_child, output.lambdaMin, output.lambdaMax
      );

      output.constraintType = oy::types::enumConstraint_t::ROTATION1D;

      return output;
   }

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintTorsionalFriction_t & constraint,
      float dt
   )
   {
      oy::types::constraintOutput_t output;
      output.jacobian = oy::constraints::torsional_friction::calculateMatrixJacobian(
         parent_stationary,
         child_stationary,
         parent_id,
         child_id,
         parent_body,
         child_body,
         constraint
      );

      output.baumgarte = oy::constraints::torsional_friction::evaluateBaumgarte(
         parent_id, child_id, parent_body, child_body, constraint, dt
      );

      oy::constraints::torsional_friction::calculateLambdaBounds(
         parent_id, child_id, parent_body, child_body, constraint, output.lambdaMin, output.lambdaMax
      );

      output.constraintType = oy::types::enumConstraint_t::TORSIONAL_FRICTION;

      return output;
   }

   oy::types::constraintOutput_t calculateConstraintOutput(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintTranslation1d_t & constraint,
      float dt
   )
   {
      oy::types::constraintOutput_t output;

      oy::types::rigidBody_t world;
      world.ql2b[0] = 1.f;

      oy::types::rigidBody_t temp_parent = (
         (parent_id != oy::types::null_body_entity)
         ? parent_body
         : world
      );

      oy::types::rigidBody_t temp_child = (
         (child_id != oy::types::null_body_entity)
         ? child_body
         : world
      );

      output.jacobian = oy::constraints::translation_1d::calculateMatrixJacobian(
         parent_stationary,
         child_stationary,
         parent_id,
         child_id,
         temp_parent,
         temp_child,
         constraint
      );

      output.baumgarte = oy::constraints::translation_1d::evaluateBaumgarte(
         parent_id, child_id, temp_parent, temp_child, constraint, dt
      );

      oy::constraints::translation_1d::calculateLambdaBounds(
         parent_id, child_id, temp_parent, temp_child, output.lambdaMin, output.lambdaMax
      );

      output.constraintType = oy::types::enumConstraint_t::TRANSLATION1D;

      return output;
   }
}
