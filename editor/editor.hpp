#ifndef EDITOR_HEADER
#define EDITOR_HEADER

#include "editor_gui.hpp"

#include "camera.hpp"
#include "default_camera_controller.hpp"
#include "viz_renderer.hpp"

#include "allocator.hpp"

class Editor
{
   public:
      Editor(void)
         : stop_sim_(false)
         , camera_controller_(camera_)
         , gui_(allocator_, renderer_)
      {
         viz::HIDInterface & temp_hid = camera_controller_;
         renderer_.setUserPointer(temp_hid);
         Vector3 camera_pos(-15.f, 0.f, 1.f);
         Vector3 camera_look_pos;
         camera_.setPos(camera_pos);
         camera_controller_.setCameraPos(camera_pos);
      }

      void loop(void);

   private:
      bool stop_sim_;

      trecs::Allocator allocator_;

      viz::Camera camera_;

      viz::DefaultCameraController camera_controller_;

      viz::VizRenderer renderer_;

      EditorGui gui_;
};

#endif
