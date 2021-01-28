#include "support_functions.hpp"

#include <cmath>

namespace geometry
{

   // Naive implementation - loops over all of the vertices in the body to find
   // the support point. A smarter version would cache vertices used previously
   // as supports.
   // Could also use an arbitrary point inside the polyhedron as a basis for
   // the support vector calculation.
   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & support_dir, const geometry::types::convexPolyhedron_t & body
   )
   {
      float max_dot = (body.verts[0] - body.center).dot(support_dir);
      geometry::types::labeledVertex_t support = {0, body.verts[0]};

      for (unsigned int i = 1; i < body.numVerts; ++i)
      {
         float dot = (body.verts[i] - body.center).dot(support_dir);
         if (dot > max_dot)
         {
            max_dot = dot;
            support.vertId = i;
            support.vert = body.verts[i];
         }
      }

      return support;
   }

   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::shapeSphere_t & body
   )
   {
      geometry::types::labeledVertex_t support;
      // Spheres don't have vertex labels.
      support.vertId = -1;
      support.vert = d.unitVector() * body.radius;

      return support;
   }

   // The capsule support is a combination of cylinder support and translated
   // sphere support.
   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::shapeCapsule_t & body
   )
   {
      // Capsule support is the support point on either spherical cap offset by
      // the cap's center point on the z-axis, unless the component of the
      // support direction in z-hat is zero.
      geometry::types::labeledVertex_t support;

      // Capsules don't have vertex labels.
      support.vertId = -1;

      Vector3 d_unit = d.unitVector();

      float cap_part = (fabs(d[2]) > 1e-7f) * (2.f * (d[2] > 0.f) - 1.f);

      support.vert = body.radius * d_unit;
      support.vert[2] += cap_part * body.height / 2;

      return support;
   }

   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::shapeCylinder_t & body
   )
   {
      geometry::types::labeledVertex_t support;

      // Cylinders don't have vertex labels.
      support.vertId = -1;

      Vector3 d_radial = d;
      d_radial[2] = 0.f;
      d_radial.Normalize();

      support.vert = body.radius * d_radial;
      support.vert[2] = (body.height / 2.f) * ((d[2] > 0.f) - (d[2] < 0.f));

      return support;
   }

   geometry::types::labeledVertex_t supportMapping(
      const Vector3 & d, const geometry::types::shapeCube_t & body
   )
   {
      geometry::types::labeledVertex_t support;

      const float default_scale = 2.f;

      float max_dot = -__FLT_MAX__;

      for (unsigned int i = 0; i < 8; ++i)
      {
         const float vert[3] = {
            (((i & 1) == 0) ? 1.f : -1.f) * body.length,
            (((i & 2) == 0) ? 1.f : -1.f) * body.width,
            (((i & 4) == 0) ? 1.f : -1.f) * body.height
         };

         float temp_dot = vert[0] * d[0] + vert[1] * d[1] + vert[2] * d[2];

         if (temp_dot > max_dot)
         {
            max_dot = temp_dot;
            support.vertId = i;
            support.vert[0] = vert[0] / default_scale;
            support.vert[1] = vert[1] / default_scale;
            support.vert[2] = vert[2] / default_scale;

         }
      }

      return support;
   }

   Matrix33 supportDirectionTransform(
      const geometry::types::transform_t & trans_A_to_W
   )
   {
      return trans_A_to_W.scale.transpose() * trans_A_to_W.rotate.transpose();
   }

   Matrix33 supportDirectionTransform(
      const geometry::types::isometricTransform_t & trans_A_to_W
   )
   {
      return trans_A_to_W.rotate.transpose();
   }

}
