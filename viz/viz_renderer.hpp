// Meshes are uploaded to the GPU in whatever frame they're received in. OpenGL
// uses y-hat as up (north-up-east) and slingshot uses z-hat as up
// (east-north-up). The type converters for model transforms assume that meshes
// are uploaded with z-hat being up, which is not what OpenGL uses.

#ifndef VIZ_RENDERER_HEADER
#define VIZ_RENDERER_HEADER

#include "camera.hpp"
#include "hid_interface.hpp"
#include "opengl_renderer.hpp"
#include "viz_types.hpp"

#include "matrix33.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"

#include "data_triangle_mesh.h"
#include "data_viz_config.h"

#include "gl_common.hpp"

#include <string>
#include <vector>

namespace viz
{
   class GuiCallbackBase;

   typedef viz::types::draw_modes::enum_t enum_draw_mode_t;

   typedef viz::types::polygon_modes::enum_t enum_polygon_mode_t;

   typedef viz::types::basic_color_t color_t;

   // A fresh face for the old viz API. It contains all of the state that was
   // previously floating around as static variables in global space.
   class VizRenderer
   {
      public:
         VizRenderer(void);

         VizRenderer(const data_vizConfig_t & config);

         ~VizRenderer(void);

         // Clears out all of the meshes in the OpenGL renderer.
         void clear(void);

         bool setUserPointer(HIDInterface & hid);

         void setWindowName(const std::string & windowName);

         // Sets the vector from the camera position to the shadow light's
         // position in FZX coordinates (z-is-up).
         void setLightDirection(const Vector3 & shadow_light_dir);

         void setAmbientLightStrength(const float ambient_light_strength);

         // Draw grid lines and all of the bodies in the renderer with no GUI
         // overlay. Returns false if rendering should stop, true otherwise.
         bool draw(Camera & camera);

         // Draw grid lines and all of the bodies in the renderer with a GUI
         // overlay specificed by the guiCallback. Returns false if rendering
         // should stop, true otherwise.
         bool draw(Camera & camera, GuiCallbackBase * guiCallback);

         // Adds a mesh to the renderer.
         int addMesh(
            const data_triangleMesh_t & mesh,
            const color_t & color,
            enum_polygon_mode_t polygonMode
         );

         int addMesh(
            const data_triangleMesh_t & mesh, enum_polygon_mode_t polygonMode
         );

         // Adds a mesh to the renderer.
         //    polygonMode = 0 --> filled triangles
         //    polygonMode = 1 --> wireframe
         int addMesh(
            const data_triangleMesh_t & mesh,
            const color_t & color,
            unsigned int polygonMode
         );

         int addMesh(const data_triangleMesh_t & mesh, unsigned int polygonMode);

         void enableMesh(unsigned int meshId);

         void disableMesh(unsigned int meshId);

         void enableShadow(unsigned int meshId);

         void disableShadow(unsigned int meshId);

         // Updates the existing vertices and triangles of a given mesh ID if
         // the meshId exists.
         int updateMesh(unsigned int meshId, const data_triangleMesh_t & mesh);

         // Updates the existing vertices and triangles and color of a mesh at
         // a given mesh ID if the meshId exists.
         int updateMesh(
            unsigned int meshId, const data_triangleMesh_t & mesh, const color_t & color
         );

         // Attempts to update an existing mesh's position, orientation, and
         // scale in the renderer by referencing its mesh ID.
         void updateMeshTransform(
            int meshId,
            const Vector3 & pos,
            const Matrix33 & R_b2l,
            const Matrix33 & scale
         );

         // Attempts to update the color of an existing mesh in the renderer by
         // referencing its mesh ID.
         void updateMeshColor(int meshId, const color_t & color);

         // Attempts to update the color of an existing mesh in the renderer by
         // referencing its mesh ID.
         void updateMeshColor(int meshId, const data_vector4_t & color);

         // Adds a line-rendered mesh to the renderer.
         int addSegment(
            unsigned int numPoints,
            const Vector3 * points,
            const color_t & color,
            enum_draw_mode_t drawMode = viz::types::draw_modes::VIZ_LINE_LOOP
         );

         // Adds a line-rendered mesh to the renderer.
         int addSegment(
            const std::vector<Vector3> & points,
            const color_t & color,
            enum_draw_mode_t drawMode = viz::types::draw_modes::VIZ_LINE_LOOP
         );

         // Updates the vertices of an existing segment with the vertices in a
         // mesh. Update is made by matching segment IDs.
         void updateSegment(
            unsigned int segmentId,
            unsigned int numPoints,
            const Vector3 * points
         );

         // Updates the vertices of an existing segment with the vertices in a
         // mesh. Update is made by matching segment IDs.
         void updateSegment(
            unsigned int segmentId,
            const std::vector<Vector3> & points
         );

         // Removes a renderable item from the renderer by referencing
         // its renderable ID.
         void deleteRenderable(unsigned int renderableId);

         void enableGrid(void);

         void disableGrid(void);

         // Returns the number of renderable items.
         unsigned int numMeshes(void) const
         {
            return ogl_renderer_.numBuffers();
         }

         unsigned int numDrawCalls(void) const
         {
            return ogl_renderer_.numDrawCalls();
         }

      private:
         // Maximum width of the GLFW window.
         const unsigned int maxInitialWidth_;

         // Maximum height of the GLFW window.
         const unsigned int maxInitialHeight_;

         // Extracts necessary information from the viz config data type.
         viz::types::config_t vizConfig_;

         GLFWwindow * vizWindow_;

         // Object that contains, draws, and transforms all of the meshes and
         // segments.
         OpenGLRenderer ogl_renderer_;

         // // Creates the OpenGL context and rendering window.
         // GLFWwindow * makeWindow(void);

         // Creates the OpenGL context and rendering window.
         GLFWwindow * makeWindow(const std::string & windowName);

         // Gives Dear ImGui all of the initial data it needs to attach to the
         // current render context.
         void initializeImgui(void);

   };
}

#endif
