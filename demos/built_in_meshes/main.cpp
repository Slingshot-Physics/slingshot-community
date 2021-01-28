#include <iostream>
#include <chrono>
#include <thread>

#include "attitudeutils.hpp"
#include "data_model.h"
#include "default_camera_controller.hpp"
#include "handle.hpp"
#include "slingshot_type_converters.hpp"
#include "geometry_type_converters.hpp"
#include "mesh.hpp"
#include "quickhull.hpp"
#include "viz_renderer.hpp"

int main(int argc, char * argv[])
{
   oy::Handle handle;
   viz::VizRenderer renderer;

   oy::types::rigidBody_t testBody;
   testBody.inertiaTensor = identityMatrix();
   testBody.mass = 1.f;
   testBody.angVel[2] = 3.f;
   testBody.linPos = Vector3(0.f, 0.f, 0.f);
   testBody.ql2b[0] = 1.f;
   geometry::types::triangleMesh_t testMesh = geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::CUBE);
   oy::types::isometricCollider_t tempCollider;
   geometry::types::shape_t tempShape;
   tempShape.shapeType = geometry::types::enumShape_t::CUBE;
   int bodyId = handle.addBody(testBody, tempCollider, tempShape, oy::types::enumRigidBody_t::DYNAMIC);

   data_triangleMesh_t meshThing;
   geometry::converters::to_pod(testMesh, &meshThing);
   int renderableId = renderer.addMesh(meshThing, 0);

   viz::Camera camera;
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   oy::types::rigidBody_t bodyStuff = handle.getBody(bodyId);

   data_triangleMesh_t temp_mesh_data;

   temp_mesh_data = geometry::mesh::loadDefaultShapeMeshData(geometry::types::enumShape_t::CYLINDER, 1.f);
   int cylId = renderer.addMesh(temp_mesh_data, 0);

   renderer.disableShadow(cylId);

   temp_mesh_data = geometry::mesh::loadDefaultShapeMeshData(geometry::types::enumShape_t::SPHERE, 1.f);
   int sphId = renderer.addMesh(temp_mesh_data, 0);

   temp_mesh_data = geometry::mesh::loadDefaultShapeMeshData(geometry::types::enumShape_t::CAPSULE, 1.f);
   int capsuleId = renderer.addMesh(temp_mesh_data, 0);

   geometry::types::shape_t shape_def;
   shape_def.shapeType = geometry::types::enumShape_t::CAPSULE;
   shape_def.capsule.radius = 3.f;
   shape_def.capsule.height = 8.f;
   geometry::types::triangleMesh_t implicitMesh = geometry::mesh::loadShapeMesh(shape_def);
   data_triangleMesh_t implicitMeshData;
   geometry::converters::to_pod(implicitMesh, &implicitMeshData);
   int implicitCapsuleId = renderer.addMesh(implicitMeshData, 0);

   shape_def.shapeType = geometry::types::enumShape_t::CUBE;
   shape_def.cube.length = 32.f;
   shape_def.cube.width = 32.f;
   shape_def.cube.height = 1.f;
   implicitMesh = geometry::mesh::loadShapeMesh(shape_def);
   geometry::converters::to_pod(implicitMesh, &implicitMeshData);
   int implicitCubeId = renderer.addMesh(implicitMeshData, 0);

   renderer.updateMeshTransform(implicitCubeId, {0.f, 0.f, -4.f}, identityMatrix(), identityMatrix());

   while (true)
   {
      handle.step();
      bodyStuff = handle.getBody(bodyId);
      Quaternion qb2l = ~bodyStuff.ql2b;

      Matrix33 R_b2l = qb2l.rotationMatrix();

      renderer.updateMeshTransform(cylId, bodyStuff.linPos + Vector3(8.0f, 0.0f, 0.0f), R_b2l, identityMatrix());
      renderer.updateMeshTransform(sphId, bodyStuff.linPos + Vector3(4.0f, 0.0f, 0.0f), R_b2l, identityMatrix());
      renderer.updateMeshTransform(capsuleId, bodyStuff.linPos + Vector3(-8.0f, 0.0f, 0.0f), R_b2l, identityMatrix());
      renderer.updateMeshTransform(renderableId, bodyStuff.linPos + Vector3(-16.0f, 0.0f, 0.0f), R_b2l, identityMatrix());
      renderer.updateMeshTransform(implicitCapsuleId, bodyStuff.linPos + Vector3(-24.0f, 0.0f, 0.0f), R_b2l, identityMatrix());
      if (renderer.draw(camera))
      {
         break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
   }

   std::cout << "Closed the window" << std::endl;
   return 0;
}
