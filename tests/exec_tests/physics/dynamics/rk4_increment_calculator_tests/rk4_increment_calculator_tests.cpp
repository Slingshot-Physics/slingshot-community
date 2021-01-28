#include "rk4_increment_calculator_system.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

// Verifies that the RK4 increment calculator works correctly for some cherry-
// picked values at the zeroth increment of RK4.
TEST_CASE( "N=0", "[Rk4IncrementCalculator]" )
{
   const int N = 0;
   trecs::Allocator allocator;
   auto rk4_increment_system = allocator.registerSystem<oy::Rk4IncrementCalculator<N> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();

   oy::types::generalizedForce_t temp_forque;
   temp_forque.appliedForce.Initialize(0.f, 0.f, 5.f);
   temp_forque.appliedTorque.Initialize(0.f, 13.f, 0.f);

   allocator.addComponent(rb_entity, temp_forque);

   oy::types::rk4Increment_t<N> temp_deriv;
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   oy::types::rk4Midpoint_t temp_rk4_midpoint;
   temp_rk4_midpoint.linPos.Initialize(0.f, 0.f, -5.f);
   temp_rk4_midpoint.linVel.Initialize(2.3f, -3.f, 3.2f);
   temp_rk4_midpoint.ql2b.Initialize(1.f, 0.f, 0.f, 0.f);
   temp_rk4_midpoint.angVel.Initialize(10.f, 0.f, 0.f);
   temp_rk4_midpoint.mass = 1.f;
   temp_rk4_midpoint.inertiaTensor = identityMatrix();

   allocator.addComponent(rb_entity, temp_rk4_midpoint);

   rk4_increment_system->update(allocator);

   const auto rk4_increment = allocator.getComponent<oy::types::rk4Increment_t<N> >(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_increment->linPosDot[i] == temp_rk4_midpoint.linVel[i] );
      REQUIRE( rk4_increment->linVelDot[i] == temp_forque.appliedForce[i] / temp_rk4_midpoint.mass );
      REQUIRE( rk4_increment->angVelDot[i] == oy::integrator::angVelDot(temp_forque.appliedTorque, temp_rk4_midpoint.angVel, temp_rk4_midpoint.inertiaTensor)[i] );
   }

   for (int i = 0; i < 4; ++i)
   {
      REQUIRE( rk4_increment->ql2bDot[i] == oy::integrator::attitudeDot(temp_rk4_midpoint.ql2b, temp_rk4_midpoint.angVel)[i] );
   }
}

TEST_CASE( "N=1", "[Rk4IncrementCalculator]" )
{
   const int N = 1;
   trecs::Allocator allocator;
   auto rk4_increment_system = allocator.registerSystem<oy::Rk4IncrementCalculator<N> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();

   oy::types::generalizedForce_t temp_forque;
   temp_forque.appliedForce.Initialize(0.f, 0.f, 5.f);
   temp_forque.appliedTorque.Initialize(0.f, 13.f, 0.f);

   allocator.addComponent(rb_entity, temp_forque);

   oy::types::rk4Increment_t<N> temp_deriv;
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   oy::types::rk4Midpoint_t temp_rk4_midpoint;
   temp_rk4_midpoint.linPos.Initialize(0.f, 0.f, -5.f);
   temp_rk4_midpoint.linVel.Initialize(2.3f, -3.f, 3.2f);
   temp_rk4_midpoint.ql2b.Initialize(1.f, 0.f, 0.f, 0.f);
   temp_rk4_midpoint.angVel.Initialize(10.f, 0.f, 0.f);
   temp_rk4_midpoint.mass = 1.f;
   temp_rk4_midpoint.inertiaTensor = identityMatrix();

   allocator.addComponent(rb_entity, temp_rk4_midpoint);

   rk4_increment_system->update(allocator);

   const auto rk4_increment = allocator.getComponent<oy::types::rk4Increment_t<N> >(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_increment->linPosDot[i] == temp_rk4_midpoint.linVel[i] );
      REQUIRE( rk4_increment->linVelDot[i] == temp_forque.appliedForce[i] / temp_rk4_midpoint.mass );
      REQUIRE( rk4_increment->angVelDot[i] == oy::integrator::angVelDot(temp_forque.appliedTorque, temp_rk4_midpoint.angVel, temp_rk4_midpoint.inertiaTensor)[i] );
   }

   for (int i = 0; i < 4; ++i)
   {
      REQUIRE( rk4_increment->ql2bDot[i] == oy::integrator::attitudeDot(temp_rk4_midpoint.ql2b, temp_rk4_midpoint.angVel)[i] );
   }
}

TEST_CASE( "N=2", "[Rk4IncrementCalculator]" )
{
   const int N = 2;
   trecs::Allocator allocator;
   auto rk4_increment_system = allocator.registerSystem<oy::Rk4IncrementCalculator<N> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();

   oy::types::generalizedForce_t temp_forque;
   temp_forque.appliedForce.Initialize(0.f, 0.f, 5.f);
   temp_forque.appliedTorque.Initialize(0.f, 13.f, 0.f);

   allocator.addComponent(rb_entity, temp_forque);

   oy::types::rk4Increment_t<N> temp_deriv;
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   oy::types::rk4Midpoint_t temp_rk4_midpoint;
   temp_rk4_midpoint.linPos.Initialize(0.f, 0.f, -5.f);
   temp_rk4_midpoint.linVel.Initialize(2.3f, -3.f, 3.2f);
   temp_rk4_midpoint.ql2b.Initialize(1.f, 0.f, 0.f, 0.f);
   temp_rk4_midpoint.angVel.Initialize(10.f, 0.f, 0.f);
   temp_rk4_midpoint.mass = 1.f;
   temp_rk4_midpoint.inertiaTensor = identityMatrix();

   allocator.addComponent(rb_entity, temp_rk4_midpoint);

   rk4_increment_system->update(allocator);

   const auto rk4_increment = allocator.getComponent<oy::types::rk4Increment_t<N> >(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_increment->linPosDot[i] == temp_rk4_midpoint.linVel[i] );
      REQUIRE( rk4_increment->linVelDot[i] == temp_forque.appliedForce[i] / temp_rk4_midpoint.mass );
      REQUIRE( rk4_increment->angVelDot[i] == oy::integrator::angVelDot(temp_forque.appliedTorque, temp_rk4_midpoint.angVel, temp_rk4_midpoint.inertiaTensor)[i] );
   }

   for (int i = 0; i < 4; ++i)
   {
      REQUIRE( rk4_increment->ql2bDot[i] == oy::integrator::attitudeDot(temp_rk4_midpoint.ql2b, temp_rk4_midpoint.angVel)[i] );
   }
}

TEST_CASE( "N=3", "[Rk4IncrementCalculator]" )
{
   const int N = 3;
   trecs::Allocator allocator;
   auto rk4_increment_system = allocator.registerSystem<oy::Rk4IncrementCalculator<N> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();

   oy::types::generalizedForce_t temp_forque;
   temp_forque.appliedForce.Initialize(0.f, 0.f, 5.f);
   temp_forque.appliedTorque.Initialize(0.f, 13.f, 0.f);

   allocator.addComponent(rb_entity, temp_forque);

   oy::types::rk4Increment_t<N> temp_deriv;
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   oy::types::rk4Midpoint_t temp_rk4_midpoint;
   temp_rk4_midpoint.linPos.Initialize(0.f, 0.f, -5.f);
   temp_rk4_midpoint.linVel.Initialize(2.3f, -3.f, 3.2f);
   temp_rk4_midpoint.ql2b.Initialize(1.f, 0.f, 0.f, 0.f);
   temp_rk4_midpoint.angVel.Initialize(10.f, 0.f, 0.f);
   temp_rk4_midpoint.mass = 1.f;
   temp_rk4_midpoint.inertiaTensor = identityMatrix();

   allocator.addComponent(rb_entity, temp_rk4_midpoint);

   rk4_increment_system->update(allocator);

   const auto rk4_increment = allocator.getComponent<oy::types::rk4Increment_t<N> >(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_increment->linPosDot[i] == temp_rk4_midpoint.linVel[i] );
      REQUIRE( rk4_increment->linVelDot[i] == temp_forque.appliedForce[i] / temp_rk4_midpoint.mass );
      REQUIRE( rk4_increment->angVelDot[i] == oy::integrator::angVelDot(temp_forque.appliedTorque, temp_rk4_midpoint.angVel, temp_rk4_midpoint.inertiaTensor)[i] );
   }

   for (int i = 0; i < 4; ++i)
   {
      REQUIRE( rk4_increment->ql2bDot[i] == oy::integrator::attitudeDot(temp_rk4_midpoint.ql2b, temp_rk4_midpoint.angVel)[i] );
   }
}
