#include "default_camera_controller.hpp"
#include "geometry.hpp"
#include "geometry_type_converters.hpp"
#include "viz_renderer.hpp"

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <thread>

#include <algorithm>
#include <set>
#include <map>

void make_cone_hull(data_triangleMesh_t & cone_hull_data)
{
   geometry::types::minkowskiDiffVertex_t cone_verts[100];

   for (unsigned int i = 0; i < 100; ++i)
   {
      cone_verts[i].bodyAVertId = -1;
      cone_verts[i].bodyBVertId = -1;
   }

   float height = 1.f;
   cone_verts[0].vert = height * Vector3(0.f, 0.f, 0.75f);

   for (unsigned int i = 0; i < 90; ++i)
   {
      cone_verts[i + 1].vert = height * Vector3(
         cosf(((float )i) * 2 * M_PI / 90), sinf(((float )i) * 2 * M_PI / 90), -0.25f
      );
   }

   geometry::types::triangleMesh_t cone_hull;
   geometry::mesh::generateHull(91, cone_verts, cone_hull);
   geometry::converters::to_pod(cone_hull, &cone_hull_data);

   std::cout << "num cone verts: " << cone_hull.numVerts << "\n";
   std::cout << "num cone triangles: " << cone_hull.numTriangles << "\n";
}

void expand_vertices(const geometry::types::triangleMesh_t &sphere_ish_mesh, std::vector<Vector3> & new_verts)
{
   float radius = sphere_ish_mesh.verts[0].magnitude();

   std::set< std::pair<unsigned int, unsigned int> > mesh_edges;
   for (unsigned int i = 0; i < sphere_ish_mesh.numTriangles; ++i)
   {
      unsigned int vert_ids[3] = {
         sphere_ish_mesh.triangles[i].vertIds[0],
         sphere_ish_mesh.triangles[i].vertIds[1],
         sphere_ish_mesh.triangles[i].vertIds[2]
      };

      std::sort(vert_ids, vert_ids + 2);

      mesh_edges.insert( std::pair<unsigned int, unsigned int>(vert_ids[0], vert_ids[1]) );
      mesh_edges.insert( std::pair<unsigned int, unsigned int>(vert_ids[1], vert_ids[2]) );
      mesh_edges.insert( std::pair<unsigned int, unsigned int>(vert_ids[0], vert_ids[2]) );
   }

   std::cout << "number of icosahedron edges: " << mesh_edges.size() << "\n";

   typedef std::set< std::pair<unsigned int, unsigned int> >::const_iterator edge_it_t;

   unsigned int num_new_mesh_verts = sphere_ish_mesh.numVerts + mesh_edges.size();

   for (edge_it_t edge_it = mesh_edges.begin(); edge_it != mesh_edges.end(); ++edge_it)
   {
      Vector3 vert0 = sphere_ish_mesh.verts[edge_it->first];
      Vector3 vert1 = sphere_ish_mesh.verts[edge_it->second];
      new_verts.push_back((0.5f * (vert0 + vert1)).Normalize() * radius);
   }

   for (unsigned int i = 0; i < sphere_ish_mesh.numVerts; ++i)
   {
      new_verts.push_back(sphere_ish_mesh.verts[i]);
   }
}

// This was only valid when icosahedra were part of the set of renderable meshes...
// I don't have the heart to delete it, but it'll cause build errors if it's still around.
// void make_sphere_hull(data_triangleMesh_t & sphere_hull_data)
// {
//    geometry::types::triangleMesh_t icosahedron_hull = geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::ICOSAHEDRON);
//    std::vector<Vector3> new_verts;

//    expand_vertices(icosahedron_hull, new_verts);

//    geometry::types::triangleMesh_t sphere1_hull;
//    geometry::mesh::generateHull(new_verts.size(), &(new_verts[0]), sphere1_hull);

//    new_verts.clear();

//    expand_vertices(sphere1_hull, new_verts);
//    geometry::types::triangleMesh_t sphere2_hull;
//    geometry::mesh::generateHull(new_verts.size(), &(new_verts[0]), sphere2_hull);

//    std::cout << "num sphere verts: " << sphere2_hull.numVerts << "\n";
//    std::cout << "num sphere triangles: " << sphere2_hull.numTriangles << "\n";

//    geometry::converters::to_pod(sphere2_hull, &sphere_hull_data);
// }

void make_capsule_hull(data_triangleMesh_t & capsule_hull_data)
{
   geometry::types::triangleMesh_t sphere = geometry::mesh::loadDefaultShapeMesh(geometry::types::enumShape_t::SPHERE);

   std::vector<Vector3> capsule_verts;
   for (unsigned int i = 0; i < sphere.numVerts; ++i)
   {
      capsule_verts.push_back(sphere.verts[i] + Vector3(0.f, 0.f, 1.f));
      capsule_verts.push_back(sphere.verts[i] + Vector3(0.f, 0.f, -1.f));
   }

   geometry::types::triangleMesh_t capsule_hull;
   geometry::mesh::generateHull(capsule_verts.size(), &(capsule_verts[0]), capsule_hull);

   std::cout << "num capsule verts: " << capsule_hull.numVerts << "\n";
   std::cout << "num capsule triangles: " << capsule_hull.numTriangles << "\n";

   geometry::converters::to_pod(capsule_hull, &capsule_hull_data);
}

int main(void)
{
   data_triangleMesh_t cone_hull_data;
   make_cone_hull(cone_hull_data);

   // write_mesh_to_filename(&cone_hull_data, "cone.mesh");

   data_triangleMesh_t sphere_hull_data = geometry::mesh::loadDefaultShapeMeshData(geometry::types::enumShape_t::SPHERE, 1.f);
   // make_sphere_hull(sphere_hull_data);

   // write_mesh_to_filename(&sphere_hull_data, "sphere.mesh");

   data_triangleMesh_t capsule_hull_data;
   make_capsule_hull(capsule_hull_data);

   // write_mesh_to_filename(&capsule_hull_data, "capsule.mesh");

   viz::VizRenderer renderer;

   viz::types::vec4_t green = {0.f, 1.f, 0.f, 1.f};
   viz::types::vec4_t red = {1.f, 0.f, 0.f, 1.f};
   viz::types::vec4_t cyan = {0.f, 1.f, 1.f, 1.f};
   viz::types::vec4_t magenta = {1.f, 0.f, 1.f, 1.f};

   viz::Camera camera;
   // renderer.initializeCamera(camera);
   viz::DefaultCameraController controller(camera);
   renderer.setUserPointer(controller);
   camera.setPos(Vector3(0, -15.0, 0.0));

   int cone_hull_id = renderer.addMesh(cone_hull_data, green, 0);
   int sphere_hull_id = renderer.addMesh(sphere_hull_data, green, 0);
   int capsule_hull_id = renderer.addMesh(capsule_hull_data, green, 0);

   while (true)
   {
      renderer.updateMeshTransform(sphere_hull_id, Vector3(15.f, 15.f, 0.f), identityMatrix(), identityMatrix());
      renderer.updateMeshTransform(capsule_hull_id, Vector3(-25.f, -25.f, 0.f), identityMatrix(), identityMatrix());
      
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (renderer.draw(camera))
      {
         break;
      }
   }

   return 0;
}
