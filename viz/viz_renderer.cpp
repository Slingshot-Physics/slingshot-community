#include "viz_renderer.hpp"

#include "callbacks.hpp"
#include "gui_callback_base.hpp"

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif

#include <vector>

namespace viz
{

   VizRenderer::VizRenderer(void)
      : maxInitialWidth_(1280)
      , maxInitialHeight_(720)
      , vizConfig_(800, 600, 60)
      , vizWindow_(makeWindow("renderer window"))
      , ogl_renderer_()
   {
      initializeImgui();
   }

   VizRenderer::VizRenderer(const data_vizConfig_t & config)
      : maxInitialWidth_(1280)
      , maxInitialHeight_(720)
      , vizConfig_(viz::converters::convert_data_to_vizConfig(config))
      , vizWindow_(makeWindow("renderer window"))
      , ogl_renderer_()
   {
      std::cout << "setup viz renderer\n";
      initializeImgui();
   }

   VizRenderer::~VizRenderer(void)
   {
      if (vizWindow_ != nullptr)
      {
         glfwTerminate();
      }
   }

   void VizRenderer::clear(void)
   {
      ogl_renderer_.clear();
   }

   bool VizRenderer::setUserPointer(HIDInterface & hid)
   {
      if (vizWindow_ == nullptr)
      {
         return false;
      }
      glfwSetWindowUserPointer(vizWindow_, static_cast<void *>(&hid));

      return true;
   }

   void VizRenderer::setWindowName(const std::string & windowName)
   {
      if (vizWindow_ != nullptr)
      {
         glfwSetWindowTitle(vizWindow_, windowName.c_str());
      }
   }

   void VizRenderer::setLightDirection(const Vector3 & light_dir_fzx)
   {
      glm::vec3 light_dir_gl;
      viz::converters::convert_fzx_to_gl(
         light_dir_fzx, light_dir_gl
      );

      ogl_renderer_.setLightDirection(light_dir_gl);
   }

   void VizRenderer::setAmbientLightStrength(const float ambient_light_strength)
   {
      ogl_renderer_.setAmbientLightStrength(ambient_light_strength);
   }

   GLFWwindow * VizRenderer::makeWindow(const std::string & windowName)
   {
      glfwInit();

      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
      glfwWindowHint(GLFW_SCALE_TO_MONITOR, GL_TRUE);

      std::cout << "Window config: " << vizConfig_.windowWidth << ", " << vizConfig_.windowHeight << "\n";

      GLFWwindow * window = glfwCreateWindow(
         std::min(vizConfig_.windowWidth, maxInitialWidth_),
         std::min(vizConfig_.windowHeight, maxInitialHeight_),
         windowName.c_str(),
         nullptr,
         nullptr
      );

      if (window == nullptr)
      {
         std::cout << "Failed to make viz window.\n";
         return window;
      }

      glfwMakeContextCurrent(window);

      glfwSetCursorPosCallback(window, viz::callbacks::mouseMove);
      glfwSetMouseButtonCallback(window, viz::callbacks::mouseButton);
      glfwSetFramebufferSizeCallback(window, viz::callbacks::frameBufferSize);
      glfwSetWindowSizeCallback(window, viz::callbacks::windowSize);
      glfwSetKeyCallback(window, viz::callbacks::keyboard);

      gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

      int frameBufferWidth, frameBufferHeight;
      glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);
      glViewport(0, 0, frameBufferWidth, frameBufferHeight);

      float xscale, yscale;
      glfwGetWindowContentScale(window, &xscale, &yscale);
      std::cout << "window scales: " << xscale << " " << yscale << std::endl;

      glEnable(GL_DEPTH_TEST);

      std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;

      // Remove the cap on frames per second
      glfwSwapInterval(0);

      return window;
   }

   void VizRenderer::initializeImgui(void)
   {
      if (vizWindow_ == nullptr)
      {
         std::cout << "Rendering window is null, can't initialize Imgui.\n";
      }

      IMGUI_CHECKVERSION();
      ImGui::CreateContext();
      ImGuiIO & io = ImGui::GetIO();
      (void)io;

      ImGui::StyleColorsDark();

      // Pilfered from demo code
      const char* glsl_version = "#version 330";
      ImGui_ImplGlfw_InitForOpenGL(vizWindow_, true);
      ImGui_ImplOpenGL3_Init(glsl_version);

      std::cout << "Finished setting up imgui\n";
   }

   bool VizRenderer::draw(Camera & camera)
   {
      return this->draw(camera, nullptr);
   }

   bool VizRenderer::draw(Camera & camera, GuiCallbackBase * guiCallback)
   {
      if (vizWindow_ != nullptr && glfwWindowShouldClose(vizWindow_))
      {
         std::cout << "Stop trying to draw!\n";
         return true;
      }
      glfwPollEvents();

      if (glfwGetKey(vizWindow_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      {
         glfwSetWindowShouldClose(vizWindow_, true);
         std::cout << "Should close" << std::endl;
         return true;
      }

      void * userPointer = glfwGetWindowUserPointer(vizWindow_);
      if (userPointer != nullptr)
      {
         viz::HIDInterface * hid = static_cast<viz::HIDInterface *>(userPointer);
         // Call the keyboard callback directly because the standard GLFW keyboard
         // callback runs at a super slow rate - this way I can guarantee that
         // this function is called at the same rate as the renderer.
         hid->keyboard_cb(vizWindow_);
      }

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      if (guiCallback != nullptr)
      {
         (*guiCallback)();
      }
      else
      {
         glClearColor(0.045f, 0.055f, 0.060f, 1.00f);
      }

      // OpenGL state-using call
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glm::mat4 view = camera.getView();
      glm::mat4 projection = camera.getProjection();

      int framebuffer_width, framebuffer_height;
      glfwGetFramebufferSize(vizWindow_, &framebuffer_width, &framebuffer_height);
      ogl_renderer_.setWindowFramebufferDims(
         static_cast<unsigned int>(framebuffer_width),
         static_cast<unsigned int>(framebuffer_height)
      );

      ogl_renderer_.setViewProjection(view, projection);
      ogl_renderer_.setCameraPos(camera.getPos());
      ogl_renderer_.setCameraLookDir(camera.getLookDir());
      ogl_renderer_.setCameraFarPlane(camera.getFarPlane());
      ogl_renderer_.draw();

      // It's important that the ImGui render call comes after the other draw
      // calls.
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(vizWindow_);

      return false;
   }

   int VizRenderer::addMesh(
      const data_triangleMesh_t & mesh,
      const color_t & color,
      enum_polygon_mode_t polygonMode
   )
   {
      int meshId = -1;

      if (mesh.numTriangles == 0 || mesh.numVerts == 0)
      {
         return meshId;
      }

      meshId = ogl_renderer_.addMesh(
         mesh,
         color.c,
         (polygonMode == enum_polygon_mode_t::VIZ_FILL) ? GL_FILL : GL_LINE
      );

      if (meshId != -1)
      {
         Vector3 defaultPos;
         updateMeshTransform(meshId, defaultPos, identityMatrix(), identityMatrix());
      }

      return meshId;
   }

   int VizRenderer::addMesh(
      const data_triangleMesh_t & mesh, enum_polygon_mode_t polygonMode
   )
   {
      return addMesh(mesh, color_t{1.f, 0.6902f, 0.0f, 1.f}, polygonMode);
   }

   int VizRenderer::addMesh(
      const data_triangleMesh_t & mesh, const color_t & color, unsigned int polygonMode
   )
   {
      enum_polygon_mode_t vizPolygonMode = (
         (polygonMode == 0) ?
         enum_polygon_mode_t::VIZ_FILL :
         enum_polygon_mode_t::VIZ_LINE
      );

      return addMesh(mesh, color, vizPolygonMode);
   }

   int VizRenderer::addMesh(const data_triangleMesh_t & mesh, unsigned int polygonMode)
   {
      return addMesh(mesh, color_t{0.5f, 0.5f, 0.5f, 1.f}, polygonMode);
   }

   void VizRenderer::enableMesh(unsigned int meshId)
   {
      ogl_renderer_.enableBuffer(meshId);
   }

   void VizRenderer::disableMesh(unsigned int meshId)
   {
      ogl_renderer_.disableBuffer(meshId);
   }

   void VizRenderer::enableShadow(unsigned int meshId)
   {
      ogl_renderer_.enableShadow(meshId);
   }

   void VizRenderer::disableShadow(unsigned int meshId)
   {
      ogl_renderer_.disableShadow(meshId);
   }

   int VizRenderer::updateMesh(unsigned int meshId, const data_triangleMesh_t & mesh)
   {
      if (!ogl_renderer_.meshValid(meshId))
      {
         return -1;
      }

      if (mesh.numTriangles == 0 || mesh.numVerts == 0)
      {
         return -1;
      }

      if (ogl_renderer_.meshSize(meshId) == 0)
      {
         ogl_renderer_.updateMesh(meshId, mesh);
         return 1;
      }

      const color_t color = ogl_renderer_.meshColor(meshId);

      ogl_renderer_.updateMesh(meshId, mesh, color.c);
      return 1;
   }

   int VizRenderer::updateMesh(
      unsigned int meshId, const data_triangleMesh_t & mesh, const color_t & color
   )
   {
      if (!ogl_renderer_.meshValid(meshId))
      {
         return -1;
      }

      ogl_renderer_.updateMesh(meshId, mesh, color.c);
      return 1;
   }

   void VizRenderer::updateMeshTransform(
      int meshId,
      const Vector3 & pos,
      const Matrix33 & R_b2l,
      const Matrix33 & scale
   )
   {
      glm::mat4 meshTrans(1.0f);
      viz::converters::convert_fzxTrans_to_GLTransM4(
         scale, R_b2l, pos, meshTrans
      );

      ogl_renderer_.updateModelTransform(meshId, meshTrans);
   }

   void VizRenderer::updateMeshColor(int meshId, const color_t & color)
   {
      glm::vec4 tempColor(color[0], color[1], color[2], color[3]);
      ogl_renderer_.updateModelColor(meshId, tempColor);
   }

   void VizRenderer::updateMeshColor(int meshId, const data_vector4_t & color)
   {
      glm::vec4 tempColor(color.v[0], color.v[1], color.v[2], color.v[3]);
      ogl_renderer_.updateModelColor(meshId, tempColor);
   }

   int VizRenderer::addSegment(
      unsigned int numPoints,
      const Vector3 * points,
      const color_t & color,
      enum_draw_mode_t drawMode
   )
   {
      int meshId = -1;

      std::vector<viz::types::basic_vertex_t> segmentVerts = \
         viz::converters::convert_Vector3_array_to_vertex_array(
            numPoints, points, color
         );

      unsigned int gl_draw_mode;
      switch(drawMode)
      {
         case viz::types::draw_modes::VIZ_TRIANGLES:
            gl_draw_mode = GL_TRIANGLES;
            break;
         case viz::types::draw_modes::VIZ_LINES:
            gl_draw_mode = GL_LINES;
            break;
         case viz::types::draw_modes::VIZ_LINE_LOOP:
            gl_draw_mode = GL_LINE_LOOP;
            break;
         default:
            gl_draw_mode = GL_TRIANGLES;
      }

      meshId = ogl_renderer_.addSegments(segmentVerts, gl_draw_mode);

      if (meshId != -1)
      {
         Vector3 defaultPos;
         updateMeshTransform(meshId, defaultPos, identityMatrix(), identityMatrix());
      }

      return meshId;
   }

   int VizRenderer::addSegment(
      const std::vector<Vector3> & points,
      const color_t & color,
      enum_draw_mode_t drawMode
   )
   {
      std::vector<viz::types::basic_vertex_t> segmentVerts = \
         viz::converters::convert_Vector3_array_to_vertex_array(
            points, color
         );

      unsigned int gl_draw_mode;
      switch(drawMode)
      {
         case viz::types::draw_modes::VIZ_TRIANGLES:
            gl_draw_mode = GL_TRIANGLES;
            break;
         case viz::types::draw_modes::VIZ_LINES:
            gl_draw_mode = GL_LINES;
            break;
         case viz::types::draw_modes::VIZ_LINE_LOOP:
            gl_draw_mode = GL_LINE_LOOP;
            break;
         default:
            gl_draw_mode = GL_TRIANGLES;
      }

      int meshId = ogl_renderer_.addSegments(segmentVerts, gl_draw_mode);

      if (meshId != -1)
      {
         Vector3 defaultPos;
         updateMeshTransform(meshId, defaultPos, identityMatrix(), identityMatrix());
      }

      return meshId;
   }

   void VizRenderer::updateSegment(
      unsigned int segmentId,
      const std::vector<Vector3> & points
   )
   {
      if (!ogl_renderer_.meshValid(segmentId))
      {
         return;
      }

      color_t color;
      if (ogl_renderer_.meshSize(segmentId) > 0)
      {
         color = ogl_renderer_.meshColor(segmentId);
      }
      else
      {
         color[0] = 0.8f;
         color[1] = 0.1f;
         color[2] = 0.4f;
         color[3] = 1.f;
      }

      std::vector<viz::types::basic_vertex_t> segmentVerts = \
         viz::converters::convert_Vector3_array_to_vertex_array(
            points, color
         );

      ogl_renderer_.updateSegments(segmentId, segmentVerts);
   }

   void VizRenderer::updateSegment(
      unsigned int segmentId,
      unsigned int numPoints,
      const Vector3 * points
   )
   {
      std::vector<Vector3> points_vec;
      points_vec.reserve(numPoints);

      for (unsigned int i = 0; i < numPoints; ++i)
      {
         points_vec.push_back(points[i]);
      }

      return updateSegment(segmentId, points_vec);
   }

   void VizRenderer::deleteRenderable(unsigned int renderableId)
   {
      ogl_renderer_.deleteRenderable(renderableId);
   }

   void VizRenderer::enableGrid(void)
   {
      ogl_renderer_.enableGrid();
   }

   void VizRenderer::disableGrid(void)
   {
      ogl_renderer_.disableGrid();
   }

}
