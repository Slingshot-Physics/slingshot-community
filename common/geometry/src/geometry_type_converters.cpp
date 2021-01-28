#include "geometry_type_converters.hpp"

#include <cstring>

namespace geometry
{

namespace converters
{

   void from_pod(
      const data_vector3_t * data_in,
      Vector3 & data_out
   )
   {
      data_out[0] = data_in->v[0];
      data_out[1] = data_in->v[1];
      data_out[2] = data_in->v[2];
   }

   void to_pod(
      const Vector3 & data_in,
      data_vector3_t * data_out
   )
   {
      data_out->v[0] = data_in[0];
      data_out->v[1] = data_in[1];
      data_out->v[2] = data_in[2];
   }

   void from_pod(
      const data_vector4_t * data_in,
      Vector4 & data_out
   )
   {
      data_out[0] = data_in->v[0];
      data_out[1] = data_in->v[1];
      data_out[2] = data_in->v[2];
   }

   void to_pod(
      const Vector4 & data_in,
      data_vector4_t * data_out
   )
   {
      data_out->v[0] = data_in[0];
      data_out->v[1] = data_in[1];
      data_out->v[2] = data_in[2];
      data_out->v[3] = data_in[3];
   }

   void from_pod(
      const data_matrix33_t * data_in,
      Matrix33 & data_out
   )
   {
      data_out(0, 0) = data_in->m[0][0];
      data_out(0, 1) = data_in->m[0][1];
      data_out(0, 2) = data_in->m[0][2];
      data_out(1, 0) = data_in->m[1][0];
      data_out(1, 1) = data_in->m[1][1];
      data_out(1, 2) = data_in->m[1][2];
      data_out(2, 0) = data_in->m[2][0];
      data_out(2, 1) = data_in->m[2][1];
      data_out(2, 2) = data_in->m[2][2];
   }

   void to_pod(
      const Matrix33 & data_in,
      data_matrix33_t * data_out
   )
   {
      data_out->m[0][0] = data_in(0, 0);
      data_out->m[0][1] = data_in(0, 1);
      data_out->m[0][2] = data_in(0, 2);
      data_out->m[1][0] = data_in(1, 0);
      data_out->m[1][1] = data_in(1, 1);
      data_out->m[1][2] = data_in(1, 2);
      data_out->m[2][0] = data_in(2, 0);
      data_out->m[2][1] = data_in(2, 1);
      data_out->m[2][2] = data_in(2, 2);
   }

   void from_pod(
      const data_isometricTransform_t * data_in,
      geometry::types::isometricTransform_t & data_out
   )
   {
      from_pod(&(data_in->rotate), data_out.rotate);
      from_pod(&(data_in->translate), data_out.translate);
   }

   void to_pod(
      const geometry::types::isometricTransform_t & data_in,
      data_isometricTransform_t * data_out
   )
   {
      to_pod(data_in.rotate, &(data_out->rotate));
      to_pod(data_in.translate, &(data_out->translate));
   }

   void from_pod(
      const data_transform_t * data_in,
      geometry::types::transform_t & data_out
   )
   {
      from_pod(&(data_in->scale), data_out.scale);
      from_pod(&(data_in->rotate), data_out.rotate);
      from_pod(&(data_in->translate), data_out.translate);
   }

   void to_pod(
      const geometry::types::transform_t & data_in,
      data_transform_t * data_out
   )
   {
      to_pod(data_in.scale, &(data_out->scale));
      to_pod(data_in.rotate, &(data_out->rotate));
      to_pod(data_in.translate, &(data_out->translate));
   }

   void from_pod(
      const data_meshTriangle_t * data_in,
      geometry::types::meshTriangle_t & data_out
   )
   {
      from_pod(&(data_in->normal), data_out.normal);
      for (unsigned int i = 0; i < 3; ++i)
      {
         data_out.vertIds[i] = data_in->vertIds[i];
      }
   }

   void to_pod(
      const geometry::types::meshTriangle_t & data_in,
      data_meshTriangle_t * data_out
   )
   {
      to_pod(data_in.normal, &(data_out->normal));
      for (unsigned int i = 0; i < 3; ++i)
      {
         data_out->vertIds[i] = data_in.vertIds[i];
      }
   }

   void from_pod(
      const data_triangle_t * data_in,
      geometry::types::triangle_t & data_out
   )
   {
      for (int i = 0; i < 3; ++i)
      {
         from_pod(&(data_in->verts[i]), data_out.verts[i]);
      }
   }

   void to_pod(
      const geometry::types::triangle_t & data_in,
      data_triangle_t * data_out
   )
   {
      for (int i = 0; i < 3; ++i)
      {
         to_pod(data_in.verts[i], &(data_out->verts[i]));
      }
   }

   void from_pod(
      const data_tetrahedron_t * data_in,
      geometry::types::tetrahedron_t & data_out
   )
   {
      for (int i = 0; i < 4; ++i)
      {
         from_pod(&(data_in->verts[i]), data_out.verts[i]);
      }
   }

   void to_pod(
      const geometry::types::tetrahedron_t & data_in,
      data_tetrahedron_t * data_out
   )
   {
      for (int i = 0; i < 4; ++i)
      {
         to_pod(data_in.verts[i], &(data_out->verts[i]));
      }
   }

   void from_pod(
      const data_shapeType_t * data_in,
      geometry::types::enumShape_t & data_out
   )
   {
      int temp = *data_in;
      data_out = (geometry::types::enumShape_t )temp;
   }

   void to_pod(
      const geometry::types::enumShape_t & data_in,
      data_shapeType_t * data_out
   )
   {
      int temp = static_cast<int>(data_in);
      *data_out = (data_shapeType_t )temp;
   }

   void from_pod(
      const data_minkowskiDiffSimplex_t * data_in,
      geometry::types::minkowskiDiffSimplex_t & data_out
   )
   {
      data_out.numVerts = data_in->numVerts;
      from_pod(&data_in->minNormBary, data_out.minNormBary);
      for (unsigned int i = 0; i < data_in->numVerts; ++i)
      {
         from_pod(&(data_in->bodyAVerts[i]), data_out.bodyAVerts[i]);
         from_pod(&(data_in->bodyBVerts[i]), data_out.bodyBVerts[i]);
         from_pod(&(data_in->verts[i]), data_out.verts[i]);
      }

      for (unsigned int i = 0; i < data_in->numVerts; ++i)
      {
         data_out.bodyAVertIds[i] = data_in->bodyAVertIds[i];
         data_out.bodyBVertIds[i] = data_in->bodyBVertIds[i];
      }
   }

   void to_pod(
      const geometry::types::minkowskiDiffSimplex_t & data_in,
      data_minkowskiDiffSimplex_t * data_out
   )
   {
      data_out->numVerts = data_in.numVerts;
      to_pod(data_in.minNormBary, &(data_out->minNormBary));
      for (unsigned int i = 0; i < data_in.numVerts; ++i)
      {
         to_pod(data_in.bodyAVerts[i], &(data_out->bodyAVerts[i]));
         to_pod(data_in.bodyBVerts[i], &(data_out->bodyBVerts[i]));
         to_pod(data_in.verts[i], &(data_out->verts[i]));
      }

      for (unsigned int i = 0; i < data_in.numVerts; ++i)
      {
         data_out->bodyAVertIds[i] = data_in.bodyAVertIds[i];
         data_out->bodyBVertIds[i] = data_in.bodyBVertIds[i];
      }
   }

   void from_pod(
      const data_polygon50_t * data_in,
      geometry::types::polygon50_t & data_out
   )
   {
      data_out.numVerts = data_in->numVerts;
      for (unsigned int i = 0; i < data_out.numVerts; ++i)
      {
         from_pod(&(data_in->verts[i]), data_out.verts[i]);
      }
   }

   void to_pod(
      const geometry::types::polygon50_t & data_in,
      data_polygon50_t * data_out
   )
   {
      data_out->numVerts = data_in.numVerts;
      for (unsigned int i = 0; i < data_out->numVerts; ++i)
      {
         to_pod(data_in.verts[i], &(data_out->verts[i]));
      }
   }

   void from_pod(
      const data_leanNeighborTriangle_t * data_in,
      geometry::types::leanNeighborTriangle_t & data_out
   )
   {
      for (int i = 0; i < 3; ++i)
      {
         data_out.neighborIds[i] = data_in->neighborIds[i];
         data_out.vertIds[i] = data_in->vertIds[i];
      }
   }

   void to_pod(
      const geometry::types::leanNeighborTriangle_t & data_in,
      data_leanNeighborTriangle_t * data_out
   )
   {
      for (int i = 0; i < 3; ++i)
      {
         data_out->neighborIds[i] = data_in.neighborIds[i];
         data_out->vertIds[i] = data_in.vertIds[i];
      }
   }

   void from_pod(
      const data_triangleMesh_t * data_in,
      geometry::types::triangleMesh_t & data_out
   )
   {
      data_out.numTriangles = data_in->numTriangles;
      data_out.numVerts = data_in->numVerts;

      for (unsigned int i = 0; i < data_in->numVerts; ++i)
      {
         from_pod(&(data_in->verts[i]), data_out.verts[i]);
      }

      for (unsigned int i = 0; i < data_in->numTriangles; ++i)
      {
         from_pod(
            &(data_in->triangles[i]),
            data_out.triangles[i]
         );
      }
   }

   void to_pod(
      const geometry::types::triangleMesh_t & data_in,
      data_triangleMesh_t * data_out
   )
   {
      data_out->numTriangles = data_in.numTriangles;
      data_out->numVerts = data_in.numVerts;

      for (unsigned int i = 0; i < data_in.numVerts; ++i)
      {
         to_pod(data_in.verts[i], &(data_out->verts[i]));
      }

      for (unsigned int i = 0; i < data_in.numTriangles; ++i)
      {
         to_pod(
            data_in.triangles[i],
            &(data_out->triangles[i])
         );
      }
   }

   void from_pod(
      const data_gaussMapFace_t * data_in,
      geometry::types::gaussMapFace_t & data_out
   )
   {
      data_out.numTriangles = data_in->numTriangles;
      data_out.triangleStartId = data_in->triangleStartId;
      from_pod(&(data_in->normal), data_out.normal);
   }

   void to_pod(
      const geometry::types::gaussMapFace_t & data_in,
      data_gaussMapFace_t * data_out
   )
   {
      data_out->numTriangles = data_in.numTriangles;
      data_out->triangleStartId = data_in.triangleStartId;
      to_pod(data_in.normal, &(data_out->normal));
   }

   void from_pod(
      const data_gaussMapMesh_t * data_in,
      geometry::types::gaussMapMesh_t & data_out
   )
   {
      data_out.numFaces = data_in->numFaces;
      data_out.numTriangles = data_in->numTriangles;
      data_out.numVerts = data_in->numVerts;

      for (unsigned int i = 0; i < data_in->numVerts; ++i)
      {
         from_pod(&(data_in->verts[i]), data_out.verts[i]);
      }

      for (unsigned int i = 0; i < data_in->numTriangles; ++i)
      {
         from_pod(&(data_in->triangles[i]), data_out.triangles[i]);
      }

      for (unsigned int i = 0; i < data_in->numFaces; ++i)
      {
         from_pod(&(data_in->faces[i]), data_out.faces[i]);
      }
   }

   void to_pod(
      const geometry::types::gaussMapMesh_t & data_in,
      data_gaussMapMesh_t * data_out
   )
   {
      data_out->numFaces = data_in.numFaces;
      data_out->numTriangles = data_in.numTriangles;
      data_out->numVerts = data_in.numVerts;

      for (unsigned int i = 0; i < data_in.numVerts; ++i)
      {
         to_pod(data_in.verts[i], &(data_out->verts[i]));
      }

      for (unsigned int i = 0; i < data_in.numTriangles; ++i)
      {
         to_pod(data_in.triangles[i], &(data_out->triangles[i]));
      }

      for (unsigned int i = 0; i < data_in.numFaces; ++i)
      {
         to_pod(data_in.faces[i], &(data_out->faces[i]));
      }
   }

   void from_pod(
      const data_shapeCapsule_t * data_in,
      geometry::types::shapeCapsule_t & data_out
   )
   {
      data_out.radius = data_in->radius;
      data_out.height = data_in->height;
   }

   void to_pod(
      const geometry::types::shapeCapsule_t & data_in,
      data_shapeCapsule_t * data_out
   )
   {
      data_out->radius = data_in.radius;
      data_out->height = data_in.height;
   }

   void from_pod(
      const data_shapeCube_t * data_in,
      geometry::types::shapeCube_t & data_out
   )
   {
      data_out.length = data_in->length;
      data_out.width = data_in->width;
      data_out.height = data_in->height;
   }

   void to_pod(
      const geometry::types::shapeCube_t & data_in,
      data_shapeCube_t * data_out
   )
   {
      data_out->length = data_in.length;
      data_out->width = data_in.width;
      data_out->height = data_in.height;
   }

   void from_pod(
      const data_shapeCylinder_t * data_in,
      geometry::types::shapeCylinder_t & data_out
   )
   {
      data_out.radius = data_in->radius;
      data_out.height = data_in->height;
   }

   void to_pod(
      const geometry::types::shapeCylinder_t & data_in,
      data_shapeCylinder_t * data_out
   )
   {
      data_out->radius = data_in.radius;
      data_out->height = data_in.height;
   }

   void from_pod(
      const data_shapeSphere_t * data_in,
      geometry::types::shapeSphere_t & data_out
   )
   {
      data_out.radius = data_in->radius;
   }

   void to_pod(
      const geometry::types::shapeSphere_t & data_in,
      data_shapeSphere_t * data_out
   )
   {
      data_out->radius = data_in.radius;
   }

   void from_pod(
      const data_shape_t * data_in,
      geometry::types::shape_t & data_out
   )
   {
      data_out.shapeType = (geometry::types::enumShape_t )data_in->shapeType;
      switch(data_out.shapeType)
      {
         case geometry::types::enumShape_t::CAPSULE:
         {
            from_pod(&(data_in->capsule), data_out.capsule);
            break;
         }
         case geometry::types::enumShape_t::CUBE:
         {
            from_pod(&(data_in->cube), data_out.cube);
            break;
         }
         case geometry::types::enumShape_t::CYLINDER:
         {
            from_pod(&(data_in->cylinder), data_out.cylinder);
            break;
         }
         case geometry::types::enumShape_t::SPHERE:
         {
            from_pod(&(data_in->sphere), data_out.sphere);
            break;
         }
         default:
         {
            break;
         }
      }
   }

   void to_pod(
      const geometry::types::shape_t & data_in,
      data_shape_t * data_out
   )
   {
      data_out->shapeType = (data_shapeType_t )data_in.shapeType;
      switch(data_out->shapeType)
      {
         case DATA_SHAPE_CAPSULE:
         {
            to_pod(data_in.capsule, &(data_out->capsule));
            break;
         }
         case DATA_SHAPE_CUBE:
         {
            to_pod(data_in.cube, &(data_out->cube));
            break;
         }
         case DATA_SHAPE_CYLINDER:
         {
            to_pod(data_in.cylinder, &(data_out->cylinder));
            break;
         }
         case DATA_SHAPE_SPHERE:
         {
            to_pod(data_in.sphere, &(data_out->sphere));
            break;
         }
         default:
         {
            break;
         }
      }
   }

   void to_pod(
      const geometry::types::testGjkInput_t & data_in,
      data_testGjkInput_t * data_out
   )
   {
      to_pod(data_in.shapeA, &(data_out->shapeA));
      to_pod(data_in.shapeB, &(data_out->shapeB));
      to_pod(data_in.transformA, &(data_out->transformA));
      to_pod(data_in.transformB, &(data_out->transformB));
   }

   void from_pod(
      const data_testGjkInput_t * data_in,
      geometry::types::testGjkInput_t & data_out
   )
   {
      from_pod(&(data_in->transformA), data_out.transformA);
      from_pod(&(data_in->shapeA), data_out.shapeA);

      from_pod(&(data_in->transformB), data_out.transformB);
      from_pod(&(data_in->shapeB), data_out.shapeB);
   }

   void to_pod(
      const geometry::types::testTetrahedronInput_t & data_in,
      data_testTetrahedronInput_t * data_out
   )
   {
      to_pod(data_in.tetrahedron, &(data_out->tetrahedron));
      to_pod(data_in.queryPointBary, &(data_out->queryPointBary));
      to_pod(data_in.queryPoint, &(data_out->queryPoint));
   }

   void from_pod(
      const data_testTetrahedronInput_t * data_in,
      geometry::types::testTetrahedronInput_t & data_out
   )
   {
      from_pod(&(data_in->tetrahedron), data_out.tetrahedron);
      from_pod(&(data_in->queryPointBary), data_out.queryPointBary);
      from_pod(&(data_in->queryPoint), data_out.queryPoint);
   }

   void to_pod(
      const geometry::types::testTriangleInput_t & data_in,
      data_testTriangleInput_t * data_out
   )
   {
      to_pod(data_in.triangle, &(data_out->triangle));
      to_pod(data_in.queryPointBary, &(data_out->queryPointBary));
      to_pod(data_in.queryPoint, &(data_out->queryPoint));
   }

   void from_pod(
      const data_testTriangleInput_t * data_in,
      geometry::types::testTriangleInput_t & data_out
   )
   {
      from_pod(&(data_in->triangle), data_out.triangle);
      from_pod(&(data_in->queryPointBary), data_out.queryPointBary);
      from_pod(&(data_in->queryPoint), data_out.queryPoint);
   }

}

}
