#include "attitudeutils.hpp"

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

TEST_CASE( "quaternions and Tait-Bryan angles" "[attitude utils]" )
{
   Vector3 initRpy(20 * M_PI / 180, 30 * M_PI / 180, -60 * M_PI / 180);
   Quaternion q;
   attitudeToQuaternion(initRpy, q);
   SECTION( "roll/pitch/yaw vector can be converted to a unit quaternion" )
   {
      REQUIRE_THAT(q.magnitude(), Catch::Matchers::WithinRel(1.f, 1e-5f));
   }
   SECTION( "rpy --> unit quat --> rpy gives the same answer" )
   {
      Vector3 outRpy;
      quaternionToAttitude(q, outRpy);
      for (int i = 0; i < 3; ++i)
      {
         REQUIRE_THAT(
            outRpy[i],
            Catch::Matchers::WithinRel(initRpy[i], 1e-5f)
         );
      }
   }
}
