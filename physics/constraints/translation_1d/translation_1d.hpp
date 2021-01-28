#ifndef TRANSLATION_1D_CONSTRAINT_HEADER
#define TRANSLATION_1D_CONSTRAINT_HEADER

#include "allocator.hpp"
#include "slingshot_types.hpp"

#include "matrix.hpp"

namespace oy
{

namespace constraints
{

namespace translation_1d
{
   void calculateLambdaBounds(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      float & lambda_min,
      float & lambda_max
   );

   // Returns a Baumgarte stabilization factor for the balljoint constraint.
   float evaluateBaumgarte(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintTranslation1d_t & constraint,
      float dt
   );

   // Returns the value of the constraint evaluated at the constrained bodies'
   // current configuration.
   float evaluateConstraint(
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintTranslation1d_t & constraint
   );

   // Removes one degree of translational freedom from one or two bodies.
   Matrix<12, 1> calculateMatrixJacobian(
      bool parent_stationary,
      bool child_stationary,
      trecs::uid_t parent_id,
      trecs::uid_t child_id,
      const oy::types::rigidBody_t & parent_body,
      const oy::types::rigidBody_t & child_body,
      const oy::types::constraintTranslation1d_t & constraint
   );

}

}

}

#endif
