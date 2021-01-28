#include "rk4_midpoint_calculator_system.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <iostream>

TEST_CASE( "N=0", "[Rk4MidpointCalculator]")
{
   trecs::Allocator allocator;
   auto rk4_midpoint_system = allocator.registerSystem<oy::Rk4MidpointCalculator<0> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();
   oy::types::rigidBody_t temp_rb;
   temp_rb.inertiaTensor = identityMatrix();
   temp_rb.ql2b[0] = 1.f;

   allocator.addComponent(rb_entity, temp_rb);

   allocator.registerComponent<oy::types::rk4Increment_t<0> >();

   oy::types::rk4Increment_t<0> temp_deriv;
   temp_deriv.angVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linPosDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   allocator.addComponent(rb_entity, oy::types::rk4Midpoint_t{});

   rk4_midpoint_system->update(allocator);

   const auto rk4_midpoint = allocator.getComponent<oy::types::rk4Midpoint_t>(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_midpoint->linPos[i] == 0.f );
      REQUIRE( rk4_midpoint->linVel[i] == 0.f );
      REQUIRE( rk4_midpoint->angVel[i] == 0.f );
   }
}

TEST_CASE( "N=1", "[Rk4MidpointCalculator]")
{
   const int N = 1;
   trecs::Allocator allocator;
   auto rk4_midpoint_system = allocator.registerSystem<oy::Rk4MidpointCalculator<N> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();
   oy::types::rigidBody_t temp_rb;
   temp_rb.inertiaTensor = identityMatrix();
   temp_rb.ql2b[0] = 1.f;

   allocator.addComponent(rb_entity, temp_rb);

   oy::types::rk4Increment_t<N - 1> temp_deriv;
   temp_deriv.angVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linPosDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   allocator.addComponent(rb_entity, oy::types::rk4Midpoint_t{});

   rk4_midpoint_system->update(allocator);

   const auto rk4_midpoint = allocator.getComponent<oy::types::rk4Midpoint_t>(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_midpoint->linPos[i] == (1.f / 1000.f) );
      REQUIRE( rk4_midpoint->linVel[i] == (1.f / 1000.f) );
      REQUIRE( rk4_midpoint->angVel[i] == (1.f / 1000.f) );
   }
}

TEST_CASE( "N=2", "[Rk4MidpointCalculator]")
{
   const int N = 2;
   trecs::Allocator allocator;
   auto rk4_midpoint_system = allocator.registerSystem<oy::Rk4MidpointCalculator<N> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();
   oy::types::rigidBody_t temp_rb;
   temp_rb.inertiaTensor = identityMatrix();
   temp_rb.ql2b[0] = 1.f;

   allocator.addComponent(rb_entity, temp_rb);

   oy::types::rk4Increment_t<N - 1> temp_deriv;
   temp_deriv.angVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linPosDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   allocator.addComponent(rb_entity, oy::types::rk4Midpoint_t{});

   rk4_midpoint_system->update(allocator);

   const auto rk4_midpoint = allocator.getComponent<oy::types::rk4Midpoint_t>(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_midpoint->linPos[i] == (1.f / 1000.f) );
      REQUIRE( rk4_midpoint->linVel[i] == (1.f / 1000.f) );
      REQUIRE( rk4_midpoint->angVel[i] == (1.f / 1000.f) );
   }
}

TEST_CASE( "N=3", "[Rk4MidpointCalculator]")
{
   const int N = 3;
   trecs::Allocator allocator;
   auto rk4_midpoint_system = allocator.registerSystem<oy::Rk4MidpointCalculator<N> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();
   oy::types::rigidBody_t temp_rb;
   temp_rb.inertiaTensor = identityMatrix();
   temp_rb.ql2b[0] = 1.f;

   allocator.addComponent(rb_entity, temp_rb);

   oy::types::rk4Increment_t<N - 1> temp_deriv;
   temp_deriv.angVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linPosDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   allocator.addComponent(rb_entity, oy::types::rk4Midpoint_t{});

   rk4_midpoint_system->update(allocator);

   const auto rk4_midpoint = allocator.getComponent<oy::types::rk4Midpoint_t>(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_midpoint->linPos[i] == (2.f / 1000.f) );
      REQUIRE( rk4_midpoint->linVel[i] == (2.f / 1000.f) );
      REQUIRE( rk4_midpoint->angVel[i] == (2.f / 1000.f) );
   }
}

TEST_CASE( "N=0 on stationary body", "[Rk4MidpointCalculator]")
{
   trecs::Allocator allocator;
   auto rk4_midpoint_system = allocator.registerSystem<oy::Rk4MidpointCalculator<0> >();

   allocator.initializeSystems();

   const auto rb_entity = allocator.addEntity();
   oy::types::rigidBody_t temp_rb;
   temp_rb.linPos.Initialize(10.f, 20.f, -30.f);
   temp_rb.linVel.Initialize(2.3f, -4.6f, 1.f);
   temp_rb.inertiaTensor = identityMatrix();
   temp_rb.ql2b[0] = 1.f;

   allocator.addComponent(rb_entity, temp_rb);

   allocator.registerComponent<oy::types::StationaryBody >();

   allocator.addComponent(rb_entity, oy::types::StationaryBody{});

   allocator.registerComponent<oy::types::rk4Increment_t<0> >();

   oy::types::rk4Increment_t<0> temp_deriv;
   temp_deriv.angVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linPosDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.linVelDot.Initialize(1.f, 1.f, 1.f);
   temp_deriv.ql2bDot.Initialize(1.f, 0.f, 0.f, 0.f);

   allocator.addComponent(rb_entity, temp_deriv);

   allocator.addComponent(rb_entity, oy::types::rk4Midpoint_t{});

   rk4_midpoint_system->update(allocator);

   const auto rk4_midpoint = allocator.getComponent<oy::types::rk4Midpoint_t>(rb_entity);

   for (int i = 0; i < 3; ++i)
   {
      REQUIRE( rk4_midpoint->linPos[i] == temp_rb.linPos[i] );
      REQUIRE( rk4_midpoint->linVel[i] == temp_rb.linVel[i] );
      REQUIRE( rk4_midpoint->angVel[i] == temp_rb.angVel[i] );
   }
}
