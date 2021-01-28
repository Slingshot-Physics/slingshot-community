#include <iostream>
#include <chrono>
#include <thread>

#include "data_model.h"
#include "default_camera_controller.hpp"
#include "geometry.hpp"
#include "geometry_type_converters.hpp"
#include "random_utils.hpp"
#include "viz_renderer.hpp"

void convert_triangle_verts_to_mesh(Vector3 * verts, data_triangleMesh_t * mesh)
{
   mesh->numVerts = 3;
   for (int i = 0; i < 3; ++i)
   {
      geometry::converters::to_pod(verts[i], &(mesh->verts[i]));
   }
   mesh->numTriangles = 1;
   mesh->triangles[0].vertIds[0] = 0;
   mesh->triangles[0].vertIds[1] = 1;
   mesh->triangles[0].vertIds[2] = 2;
}

void convert_tetrahedron_verts_to_mesh(Vector3 * verts, data_triangleMesh_t * mesh)
{
   mesh->numVerts = 4;
   for (int i = 0; i < 4; ++i)
   {
      geometry::converters::to_pod(verts[i], &(mesh->verts[i]));
   }

   mesh->numTriangles = 4;
   mesh->triangles[0].vertIds[0] = 0;
   mesh->triangles[0].vertIds[1] = 1;
   mesh->triangles[0].vertIds[2] = 2;

   mesh->triangles[1].vertIds[0] = 1;
   mesh->triangles[1].vertIds[1] = 2;
   mesh->triangles[1].vertIds[2] = 3;

   mesh->triangles[2].vertIds[0] = 0;
   mesh->triangles[2].vertIds[1] = 1;
   mesh->triangles[2].vertIds[2] = 3;

   mesh->triangles[3].vertIds[0] = 0;
   mesh->triangles[3].vertIds[1] = 2;
   mesh->triangles[3].vertIds[2] = 3;

   // Ignoring normals, I think that's fine?
}

int main(int argc, char * argv[])
{

   viz::VizRenderer renderer;

   glDisable(GL_CULL_FACE);

   const unsigned int numTriangles = 32;

   Vector3 triangleCenters[numTriangles];
   Vector3 tetrahedronCenters[numTriangles];
   Vector3 randoPoints[numTriangles];

   for (unsigned int i = 0; i < numTriangles; ++i)
   {
      triangleCenters[i][0] = 2 * ((float )(i) - ((float )numTriangles)/2.0f);
      tetrahedronCenters[i][1] = -6.0f;
      tetrahedronCenters[i][0] = 2 * ((float )(i) - ((float )numTriangles)/2.0f);
      randoPoints[i] = edbdmath::random_vec3(0.f, 0.5f);
   }

   geometry::types::triangle_t triangles[numTriangles];
   geometry::types::tetrahedron_t tetrahedrons[numTriangles];

   viz::types::basic_color_t triangleColors = {{0.0f, 1.0f, 0.1f, 0.3f}};
   viz::types::basic_color_t simplexColors = {{0.0f, 0.10f, 1.0f, 1.0f}};
   viz::types::basic_color_t minNormDirColors = {{0.7f, 0.10f, 0.7f, 1.0f}};
   viz::types::basic_color_t randoPointColors = {{1.0f, 0.00f, 0.1f, 1.0f}};

   data_triangleMesh_t tempMesh;
   int tempMeshId;
   for (unsigned int i = 0; i < numTriangles; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         triangles[i].verts[j] = edbdmath::random_vec3(0.f, 1.5f);
      }
      convert_triangle_verts_to_mesh(triangles[i].verts, &tempMesh);
      tempMeshId = renderer.addMesh(tempMesh, triangleColors, 0);
      renderer.updateMeshTransform(tempMeshId, triangleCenters[i], identityMatrix(), identityMatrix());

      if (i == 0)
      {
         tetrahedrons[0].verts[0] = Vector3(1.0f, 2.0f, -2.0f);
         tetrahedrons[0].verts[1] = Vector3(-2.0f, -1.0f, -3.0f);
         tetrahedrons[0].verts[2] = Vector3(1.0f, 0.0f, -2.0f);
         tetrahedrons[0].verts[3] = Vector3(-1.0f, 2.0f, -2.0f);
      }
      else
      {
         for (int j = 0; j < 4; ++j)
         {
            tetrahedrons[i].verts[j] = edbdmath::random_vec3(0.f, 2.5f);
         }
      }

      convert_tetrahedron_verts_to_mesh(tetrahedrons[i].verts, &tempMesh);

      tempMeshId = renderer.addMesh(tempMesh, triangleColors, 0);
      renderer.updateMeshTransform(tempMeshId, tetrahedronCenters[i], identityMatrix(), identityMatrix());
   }

   randoPoints[0] = Vector3();

   geometry::types::triangle_t plotVecs[numTriangles];
   geometry::types::triangle_t tetraPlotVecs[numTriangles];
   Vector3 closestPoints[numTriangles];
   Vector3 tetraClosestPoints[numTriangles];

   for (unsigned int i = 0; i < numTriangles; ++i)
   {
      closestPoints[i] = geometry::triangle::closestPointToPoint(
         triangles[i].verts[0], triangles[i].verts[1], triangles[i].verts[2], randoPoints[i]
      ).point;
      plotVecs[i].verts[0] = closestPoints[i];
      plotVecs[i].verts[1] = randoPoints[i];

      tempMeshId = renderer.addSegment(2, plotVecs[i].verts, randoPointColors);
      renderer.updateMeshTransform(tempMeshId, triangleCenters[i], identityMatrix(), identityMatrix());

      tetraClosestPoints[i] = geometry::tetrahedron::closestPointToPoint(
         tetrahedrons[i].verts[0],
         tetrahedrons[i].verts[1],
         tetrahedrons[i].verts[2],
         tetrahedrons[i].verts[3],
         randoPoints[i]
      ).point;

      tetraPlotVecs[i].verts[0] = tetraClosestPoints[i];
      tetraPlotVecs[i].verts[1] = randoPoints[i];

      tempMeshId = renderer.addSegment(2, tetraPlotVecs[i].verts, randoPointColors);
      renderer.updateMeshTransform(tempMeshId, tetrahedronCenters[i], identityMatrix(), identityMatrix());

   }

   Vector3 mover[2];
   mover[0][0] = 1.0f;
   mover[0][1] = 0.0f;
   mover[0][2] = -2.0f;

   mover[1][0] = -1.0f;
   mover[1][1] = 0.0f;
   mover[1][2] = 2.0f;

   float t = 0.0f;

   unsigned int moverId = renderer.addSegment(2, mover, randoPointColors);

   viz::Camera camera;
   // renderer.initializeCamera(camera);
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   Quaternion qb2l(1.0f, 0.0f, 0.0f, 0.0f);

   unsigned int loop = 0;
   unsigned int meshIdToDelete = 4 * numTriangles - 1;
   while(true)
   {
      mover[0][0] = cosf(0.25 * t);
      mover[0][2] = -2.0f * sinf(0.25 * t);
      mover[1][0] = -1.0f * cosf(0.25 * t);
      mover[1][2] = 2.0f * sinf(0.25 * t);

      renderer.updateSegment(moverId, 2, mover);

      if ((loop + 1) % 100 == 0)
      {
         std::cout << loop << "\n";
      }

      if ((loop + 1) % 50 == 0)
      {
         std::cout << "deleting a mesh!\n";
         renderer.deleteRenderable(meshIdToDelete);
         meshIdToDelete -= 1;
      }

      if (renderer.draw(camera))
      {
         break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(5));

      t += 0.1f;

      ++loop;
   }

   std::cout << "Closed the window" << std::endl;
   return 0;
}
