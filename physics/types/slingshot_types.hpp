#ifndef FZX_TYPES_HEADER
#define FZX_TYPES_HEADER

#include "data_model.h"
#include "ecs_types.hpp"
#include "epa_types.hpp"
#include "geometry_types.hpp"
#include "logger_types.hpp"
#include "matrix.hpp"
#include "matrix33.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"

#include <map>
#include <utility>
#include <vector>

#define MAX_VERTICES 500

namespace oy
{

namespace types
{
   typedef geometry::types::shapeCube_t cube_t;
   typedef geometry::types::shapeSphere_t sphere_t;
   typedef geometry::types::shapeCapsule_t capsule_t;
   typedef geometry::types::shapeCylinder_t cylinder_t;

   // This is used as a body ID when a constraint or force is tied to the world
   // instead of a rigid body. The `null_body_id` should be used in scenario
   // definitions (e.g. JSON files), not the simulation.
   const int null_body_id = -1;

   // This is used as a body entity when a constraint or force is tied to the
   // world instead of a rigid body. The `null_body_entity` should be used in
   // the simulation.
   const trecs::uid_t null_body_entity = -1;

   enum class enumFrame_t
   {
      BODY = 0,
      GLOBAL = 1,
   };

   enum class enumConstraint_t
   {
      NONE = 0,
      BALLJOINT = 1,
      COLLISION = 2,
      FRICTION = 3,
      REVOLUTEJOINT = 4,
      REVOLUTEMOTOR = 6,
      GEAR = 7,
      TRANSLATION1D = 8,
      ROTATION1D = 9,
      TORSIONAL_FRICTION = 10,
   };

   enum class enumRigidBody_t
   {
      STATIONARY = 0,
      DYNAMIC = 1
   };

   enum class enumRaycastBodyFilter_t
   {
      ALL = 0,
      DYNAMIC = 1,
      STATIONARY = 2,
   };

   struct orientedMass_t
   {
      float mass;

      // Moment of inertia tensor measured about the body's CM. This is
      // typically the principal moments of inertia.
      Matrix33 inertiaTensor;

      Vector3 linPos;

      // Unit quaternion to rotate lab to body frame.
      Quaternion ql2b;
   };

   struct generalizedVelocity_t
   {
      Vector3 linVel;

      // Angular velocity in body frame.
      Vector3 angVel;
   };

   struct generalizedForce_t
   {
      // Applied force in world coordinates.
      Vector3 appliedForce;

      // Applied torque in body frame.
      Vector3 appliedTorque;
   };

   // A component tag that marks a rigid body as stationary. This allows
   // systems to directly generate queries for stationary body types rather
   // than grabbing all body types and checking fields.
   struct StationaryBody
   { };

   // A component tag that marks a rigid body as dynamic. This allows
   // systems to directly generate queries for dynamic body types rather
   // than grabbing all body types and checking fields.
   struct DynamicBody
   { };

   struct rigidBody_t
   {
      float mass;
      Vector3 linPos;
      Vector3 linVel;
      // Angular velocity is in body frame.
      Vector3 angVel;
      Matrix33 inertiaTensor;
      // Rotate lab to body frame
      Quaternion ql2b;
   };

   // A midpoint in the RK4 algorithm,
   //   x_mid = x_n + dt * k_i
   // where k_i is an increment of RK4 and x_n is a rigid body state at time n.
   struct rk4Midpoint_t
   {
      Vector3 linPos;
      Vector3 linVel;
      // Rotate lab to body frame
      Quaternion ql2b;
      // Angular velocity is in body frame.
      Vector3 angVel;
      float mass;
      Matrix33 inertiaTensor;
   };

   // An individual increment out of RK4. The time derivative of the elements
   // of a rigid body's states.
   template <int N>
   struct rk4Increment_t
   {
      Vector3 linPosDot;
      Vector3 linVelDot;
      Quaternion ql2bDot;
      Vector3 angVelDot;
   };

   struct constrainedRigidBody_t
   {
      // The body's inverse inertia tensor in world coordinates.
      Matrix33 invInertia;

      // The quantity
      //    nextQVel = qVel + qAccel * dt
      // in world coordinates. The first three elements are linear velocity,
      // the last three elements are angular velocity.
      Matrix<6, 1> nextQVel;

      // Inverse mass scalar.
      float invMass;
   };

   // The definition of a rigid body's convex hull.
   struct isometricCollider_t
   {
      // The coefficient of restitution for this body.
      float restitution;

      // The coefficient of friction for this body.
      float mu;

      // If enabled, any other body with an enabled collider will be able to
      // collide with this body.
      bool enabled;
   };

   struct multiCollider_t
   {
      unsigned int numColliders;
      isometricCollider_t colliders[64];
   };

   struct compositeConstraintJoint_t
   {
      unsigned int numRotationConstraints;

      trecs::uid_t rotationConstraintIds[3];

      unsigned int numTranslationConstraints;

      trecs::uid_t translationConstraintIds[3];
   };

   struct constraintBalljoint_t
   {
      // The link point's position in the parent's CM-centered body frame.
      Vector3 parentLinkPoint;

      // The link point's position in the child's CM-centered body frame.
      Vector3 childLinkPoint;
   };

   // All kinematic quantities are in world coordinates (ENU)
   struct constraintCollision_t
   {
      trecs::uid_t bodyIdA;
      trecs::uid_t bodyIdB;

      // Collision normal as a vector emanating from body A.
      Vector3 unitNormal;

      // Contact point on the body in global coordinates relative to the body's
      // center of mass in global coordinates.
      Vector3 bodyAContact;

      // Contact point on the body in global coordinates relative to the body's
      // center of mass in global coordinates.
      Vector3 bodyBContact;

      float restitution;
   };

   struct constraintFriction_t
   {
      trecs::uid_t bodyIdA;
      trecs::uid_t bodyIdB;

      // Total coefficient of friction between the two bodies.
      float muTotal;

      // Total force applied on body A in world coordinates.
      Vector3 bodyAForce;

      // Total force applied on body B in world coordinates.
      Vector3 bodyBForce;

      // Contact point on body A in world coordinates.
      Vector3 bodyAContact;

      // Contact point on body B in world coordinates.
      Vector3 bodyBContact;

      // The contact normal emanating from body A.
      Vector3 unitNormal;
   };

   struct constraintGear_t
   {
      // Radius of the hypothetical gear on the parent body.
      float parentGearRadius;

      // Radius of the hypothetical gear on the child body.
      float childGearRadius;

      // Axis of rotation for the parent body in its CM-centered body frame.
      Vector3 parentAxis;

      // Axis of rotation for the child body in its CM-centered body frame.
      Vector3 childAxis;

      // If this is true, the signs of the body frame angular velocities dotted
      // with the body frame rotation axes will be the same. If false, the
      // signs will be the opposite.
      bool rotateParallel;
   };

   struct constraintRevoluteJoint_t
   {
      // The link points' positions in the parent's CM-centered body frame.
      Vector3 parentLinkPoints[2];

      // The link points' positions in the child's CM-centered body frame.
      Vector3 childLinkPoints[2];
   };

   struct constraintRevoluteMotor_t
   {
      // The motor's axis of rotation in the parent's CM-centered body frame.
      Vector3 parentAxis;

      // The motor's axis of rotation in the child's CM-centered body frame.
      Vector3 childAxis;

      // The motor's angular speed set-point along the axis of rotation in one
      // coordinate frame (scalars are coordinate frame invaraint).
      float angularSpeed;

      // The maximum torque the motor can exert between the bodies.
      float maxTorque;
   };

   // The parent and child axes are constrained to be orthogonal to each other.
   struct constraintRotation1d_t
   {
      // A unit vector in the parent's body frame.
      Vector3 parentAxis;

      // A unit vector in the child's body frame.
      Vector3 childAxis;
   };

   struct constraintTorsionalFriction_t
   {
      trecs::uid_t bodyIdA;
      trecs::uid_t bodyIdB;

      // Total coefficient of friction between the two bodies.
      float muTotal;

      // Max distance between the contact points and the center of the contact
      // points. Used to calculate the lambda min/max bounds.
      float leverArmLength;

      // Total force applied on body A in world coordinates.
      Vector3 bodyAForce;

      // Total force applied on body B in world coordinates.
      Vector3 bodyBForce;

      // The contact normal emanating from body A.
      Vector3 unitNormal;
   };

   // The child's link point is constrained to move around the plane defined by
   // the parent's link point and the parent's axis.
   struct constraintTranslation1d_t
   {
      // A vector in the parent's body frame defining the normal of the
      // constraint plane in the parent's body frame.
      Vector3 parentAxis;

      // A point in the parent's body frame that is on the constraint plane in
      // the parent's body frame.
      Vector3 parentLinkPoint;

      // A point in the child's body frame that is constrained to live on the
      // plane defined in the parent's body frame.
      Vector3 childLinkPoint;
   };

   struct raycastResult_t
   {
      // The ID of the body that was hit by the ray. Only valid if 'hit'
      // is 'true'.
      trecs::uid_t bodyId;

      // True if the ray hit something, false otherwise.
      bool hit;

      // The number of intersections the ray makes with 'bodyId', if 'hit'
      // is 'true'.
      unsigned int numHits;

      // The locations of the hit points on 'bodyId'. The hit vector at the
      // lowest index is closer to the ray start position than the hit vector
      // at the highest index.
      Vector3 hits[2];
   };

   // Tracks metadata for inequality and equality constraint types in the
   // constraint solver, used as an edge identifier in a graph edge.
   struct edgeId_t
   {
      // The ECS entity of the constraint or vector of constraints. This is
      // the entity of a single equality constraint or an entity of a vector
      // of inequality constraints.
      trecs::uid_t entity;

      // This tells the constraint solver the index of the constraint in a
      // vector of inequality constraints.
      int offset;

      enumConstraint_t constraintType;
   };

   struct forceConstant_t
   {
      // The point where the constant force is applied in the child's
      // CM-centered body frame.
      Vector3 childLinkPoint;

      // The acceleration applied to the body.
      Vector3 acceleration;

      // The frame the constant acceleration is applied in.
      enumFrame_t forceFrame;
   };

   struct forceSpring_t
   {
      // The length at which the spring exerts zero force.
      float restLength;
      // Spring coefficient, should be less than zero.
      float springCoeff;
      // The link point's position in the parent's CM-centered body frame.
      Vector3 parentLinkPoint;
      // The link point's position in the child's CM-centered body frame.
      Vector3 childLinkPoint;
   };

   struct forceVelocityDamper_t
   {
      // Damper coefficient, should be less than zero.
      float damperCoeff;
      // The link point's position in the parent's CM-centered body frame.
      Vector3 parentLinkPoint;
      // The link point's position in the child's CM-centered body frame.
      Vector3 childLinkPoint;
   };

   struct forceDrag_t
   {
      // Drag coefficient for velocity, should be less than or equal to zero.
      float linearDragCoeff;

      // Drag coefficient for quadratic velocity, should be less than or equal
      // to zero.
      float quadraticDragCoeff;
   };

   struct torqueDrag_t
   {
      // Drag coefficient for angular velocity, should be less than or equal to
      // zero.
      float linearDragCoeff;

      // Drag coefficient for quadratic angular velocity, should be less than
      // or equal to zero.
      float quadraticDragCoeff;
   };

   template <typename ShapeA_T, typename ShapeB_T>
   struct shapeTuple_t
   {
      ShapeA_T shapeA;
      ShapeB_T shapeB;
   };

   struct transformTuple_t
   {
      // Forward transform that converts from body A's body coordinates to
      // world coordinates
      geometry::types::transform_t trans_A_to_W;

      // Forward transform that converts from body B's body coordinates to
      // world coordinates
      geometry::types::transform_t trans_B_to_W;
   };

   struct contactGeometry_t
   {
      void initialize(const geometry::types::epaResult_t & epa_out)
      {
         contactNormal = epa_out.p;
         bodyAContactPoint = epa_out.bodyAContactPoint;
         bodyBContactPoint = epa_out.bodyBContactPoint;
      }

      Vector3 contactNormal;
      Vector3 bodyAContactPoint;
      Vector3 bodyBContactPoint;
   };

   template <typename ShapeA_T, typename ShapeB_T>
   struct collisionCandidate_t
   {
      int nodeIdA;
      int nodeIdB;

      // Forward transform that converts from body A's body coordinates to
      // world coordinates
      geometry::types::isometricTransform_t trans_A_to_W;

      // Forward transform that converts from body B's body coordinates to
      // world coordinates
      geometry::types::isometricTransform_t trans_B_to_W;

      ShapeA_T shape_a;

      ShapeB_T shape_b;
   };

   template <typename ShapeA_T, typename ShapeB_T>
   struct collisionDetection_t
   {
      inline void initialize(
         const collisionCandidate_t<ShapeA_T, ShapeB_T> & candidate,
         const geometry::types::minkowskiDiffSimplex_t & md_simplex
      )
      {
         nodeIdA = candidate.nodeIdA;
         nodeIdB = candidate.nodeIdB;
         trans_A_to_W = candidate.trans_A_to_W;
         trans_B_to_W = candidate.trans_B_to_W;
         shape_a = candidate.shape_a;
         shape_b = candidate.shape_b;
         simplex = md_simplex;
      }

      int nodeIdA;
      int nodeIdB;

      // Forward transform that converts from body A's body coordinates to
      // world coordinates
      geometry::types::isometricTransform_t trans_A_to_W;

      // Forward transform that converts from body B's body coordinates to
      // world coordinates
      geometry::types::isometricTransform_t trans_B_to_W;

      ShapeA_T shape_a;

      ShapeB_T shape_b;

      geometry::types::minkowskiDiffSimplex_t simplex;
   };

   template <typename ShapeA_T, typename ShapeB_T>
   struct collisionGeometry_t
   {
      inline void initialize(
         const collisionDetection_t<ShapeA_T, ShapeB_T> & detection,
         const geometry::types::epaResult_t & epa_result
      )
      {
         nodeIdA = detection.nodeIdA;
         nodeIdB = detection.nodeIdB;
         trans_A_to_W = detection.trans_A_to_W;
         trans_B_to_W = detection.trans_B_to_W;
         shape_a = detection.shape_a;
         shape_b = detection.shape_b;
         contactNormal = epa_result.p;
         bodyAContactPoint = epa_result.bodyAContactPoint;
         bodyBContactPoint = epa_result.bodyBContactPoint;
      }

      inline void initialize(
         const collisionCandidate_t<ShapeA_T, ShapeB_T> & candidate,
         const geometry::types::satResult_t & sat_result
      )
      {
         nodeIdA = candidate.nodeIdA;
         nodeIdB = candidate.nodeIdB;
         trans_A_to_W = candidate.trans_A_to_W;
         trans_B_to_W = candidate.trans_B_to_W;
         shape_a = candidate.shape_a;
         shape_b = candidate.shape_b;
         contactNormal = sat_result.contactNormal;
         bodyAContactPoint = sat_result.deepestPointsA[0];
         bodyBContactPoint = sat_result.deepestPointsB[0];
      }

      int nodeIdA;
      int nodeIdB;

      // Forward transform that converts from body A's body coordinates to
      // world coordinates
      geometry::types::isometricTransform_t trans_A_to_W;

      // Forward transform that converts from body B's body coordinates to
      // world coordinates
      geometry::types::isometricTransform_t trans_B_to_W;

      ShapeA_T shape_a;

      ShapeB_T shape_b;

      // Vector of maximum penetration from body A to body B (world coords).
      Vector3 contactNormal;

      // One of the points on body A that penetrates most deeply into body B
      // in world coordinates. Points of deepest contact may be degenerate.
      Vector3 bodyAContactPoint;

      // One of the points on body B that penetrates most deeply into body A
      // in world coordinates. Points of deepest contact may be degenerate.
      Vector3 bodyBContactPoint;
   };

   // A reduced collision contact manifold between two bodies. A minimum of
   // four contact points are required for stable contact.
   struct collisionContactManifold_t
   {
      // Collision normal as a vector emanating from body A.
      Vector3 unitNormal;

      trecs::uid_t bodyIdA;
      trecs::uid_t bodyIdB;

      // The reduced contact manifold of points on body A.
      Vector3 bodyAContacts[4];

      // The reduced contact manifold of points on body B.
      Vector3 bodyBContacts[4];

      // Indicates the number of contact points interacting between bodies A
      // and B, between [0, 4].
      unsigned int numContacts;

      float bodyARestitution;
      float bodyBRestitution;

      float bodyAMu;
      float bodyBMu;
   };

   // Used for retaining data related to constraint solutions.
   struct constraintOutput_t
   {
      // The constraint Jacobian in world coordinates
      Matrix<12, 1> jacobian;

      // Baumgarte factor
      float baumgarte;

      // Minimum and maximum values for the Lagrange multiplier calculated
      // by Projected Gauss Seidel
      float lambdaMin;
      float lambdaMax;

      enumConstraint_t constraintType;
   };

   // Disambiguates the inequality constraint output pipelines between each
   // inequality constraint type.
   template <typename Constraint_T>
   struct categoryInequalityConstraint_t
   { };

   // Used in the scenario definition to manage connections between rigid
   // bodies by constraints or forces.
   struct bodyLink_t
   {
      trecs::uid_t parentId;
      trecs::uid_t childId;
   };

   // Internal definition of a scenario, used to initialize the simulation.
   // Maps ID's are provided by the scenario data type when converting from a
   // data model type, and map ID's are the internal ID's when pulling data
   // from the allocator.
   struct scenario_t
   {
      std::map<int, isometricCollider_t> isometric_colliders;

      std::map<int, rigidBody_t> bodies;

      std::map<int, enumRigidBody_t> body_types;

      std::map<int, geometry::types::shape_t> shapes;

      std::vector<std::pair<bodyLink_t, constraintBalljoint_t> > balljoints;

      std::vector<std::pair<bodyLink_t, constraintGear_t> > gears;

      std::vector<std::pair<bodyLink_t, constraintRevoluteJoint_t> > revolute_joints;

      std::vector<std::pair<bodyLink_t, constraintRevoluteMotor_t> > revolute_motors;

      std::vector<std::pair<bodyLink_t, constraintRotation1d_t> > rotation_1d;

      std::vector<std::pair<bodyLink_t, constraintTranslation1d_t> > translation_1d;

      std::vector<std::pair<bodyLink_t, forceConstant_t> > constant_forces;

      std::vector<std::pair<bodyLink_t, forceDrag_t> > drag_forces;

      std::vector<std::pair<bodyLink_t, forceSpring_t> > springs;

      std::vector<std::pair<bodyLink_t, forceVelocityDamper_t> > dampers;

      std::vector<std::pair<bodyLink_t, torqueDrag_t> > drag_torques;

      logger::types::loggerConfig_t logger;

      void clear(void)
      {
         bodies.clear();
         shapes.clear();
         balljoints.clear();
         gears.clear();
         revolute_joints.clear();
         revolute_motors.clear();
         translation_1d.clear();
         rotation_1d.clear();
         drag_forces.clear();
         constant_forces.clear();
         springs.clear();
         dampers.clear();
         drag_torques.clear();
      }
   };

} // types

} // fzx

#endif
