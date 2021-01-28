#include "transform_utils.hpp"

#include "attitudeutils.hpp"
#include "random_utils.hpp"

namespace test_utils
{

   // Gernerates a roll, pitch, yaw vector and converts that into a rotation
   // matrix from FRD to NED.
   geometry::types::transform_t generate_random_transform(void)
   {
      geometry::types::transform_t rando_trans;
      rando_trans.scale = identityMatrix();
      rando_trans.translate = edbdmath::random_vec3(-2.f, 2.f);
      Vector3 rpy = edbdmath::random_vec3(
         -2.f * M_PI, 2.f * M_PI, -M_PI, M_PI, -2.f * M_PI, 2.f * M_PI
      );

      // Turn the RPY sequence into a FRD to NED matrix.
      rando_trans.rotate = frd2NedMatrix(rpy);

      return rando_trans;
   }

   geometry::types::transform_t generate_random_scale_transform(void)
   {
      geometry::types::transform_t rando_trans;

      rando_trans.scale.diag(edbdmath::random_vec3(0.1f, 3.f));
      rando_trans.translate = edbdmath::random_vec3(-2.f, 2.f);
      Vector3 rpy = edbdmath::random_vec3(
         -2.f * M_PI, 2.f * M_PI, -M_PI, M_PI, -2.f * M_PI, 2.f * M_PI
      );

      // Turn the RPY sequence into a FRD to NED matrix.
      rando_trans.rotate = frd2NedMatrix(rpy);

      return rando_trans;
   }

   void convert_triangleMesh_to_convexPolyhedron(
      const geometry::types::triangleMesh_t & data_in,
      geometry::types::convexPolyhedron_t & data_out
   )
   {
      for (unsigned int i = 0; i < data_in.numVerts; ++i)
      {
         data_out.verts[i] = data_in.verts[i];
      }

      data_out.numVerts = data_in.numVerts;
      data_out.center = Vector3(0.f, 0.f, 0.f);
      for (unsigned int i = 0; i < data_out.numVerts; ++i)
      {
         data_out.center += data_in.verts[i];
      }
      data_out.center /= (float )data_out.numVerts;
   }

}
