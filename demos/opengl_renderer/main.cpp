#include <iostream>
#include <chrono>
#include <thread>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include "attitudeutils.hpp"
#include "data_model.h"
#include "default_camera_controller.hpp"
#include "geometry.hpp"
#include "viz_renderer.hpp"

int main(int argc, char * argv[])
{
   viz::VizRenderer renderer;

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   data_triangleMesh_t sphere_data = geometry::mesh::loadDefaultShapeMeshData(geometry::types::enumShape_t::SPHERE, 1.f);

   viz::types::vec4_t red = {1.0f, 0.0f, 0.0f, 1.0f};
   viz::types::vec4_t blue = {0.0f, 0.0f, 1.0f, 1.0f};

   int sphereMeshId = renderer.addMesh(sphere_data, blue, 0);

   Vector3 mesh1Offset(0.0f, 0.0f, 0.0f);

   renderer.updateMeshTransform(sphereMeshId, mesh1Offset, identityMatrix(), identityMatrix());

   int sphereMeshId2 = renderer.addMesh(sphere_data, red, 0);
   Vector3 mesh2Offset(0.0f, 0.0f, 5.0f);
 
   std::cout << "mesh IDs: " << sphereMeshId << ", " << sphereMeshId2 << "\n";

   renderer.updateMeshTransform(sphereMeshId2, mesh2Offset, identityMatrix(), identityMatrix());

   int i = 0;
   while(true)
   {
      if (renderer.draw(camera))
      {
         break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(5));

      if (i % 100 == 0)
      {
         std::cout << (i / 500.0f) << "\n";
      }

      mesh2Offset = Vector3(cosf(i * M_PI/50), 0.0f, 5.0f + sinf(i * M_PI/50));

      renderer.updateMeshTransform(sphereMeshId2, mesh2Offset, identityMatrix(), identityMatrix());
      i += 1;
   }

   std::cout << "Closed the window" << std::endl;
   return 0;
}
