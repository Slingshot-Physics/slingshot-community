#ifndef CAMERA_HEADER
#define CAMERA_HEADER

// This is necessary for Mac - the latest version of GLM deprecates calls with
// radians.
#define GLM_FORCE_RADIANS

#include "vector3.hpp"
#include "viztypeconverters.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace viz
{

   // The camera class contains a position, look direction, up vector, and
   // projection matrix for draw calls. All calls to transform setters use FZX
   // coordinates where z-hat is up. This class explicitly transforms vectors
   // from FZX coordinates to OpenGL coordinates.
   class Camera
   {
      public:
         Camera(void)
            : far_plane_(64.f)
         {
            viz::types::cameraTransform_t cameraTransform;
            cameraTransform.pos = Vector3(5.0f, 5.0f, 15.0f);
            cameraTransform.look_direction = Vector3(0.0f, 1.0f, 0.0f);
            cameraTransform.up = Vector3(0.0f, 0.0f, 1.0f);

            initialize(cameraTransform, 800, 600);
         }

         // Set the camera's transform in FZX coordinates.
         void setTransform(const viz::types::cameraTransform_t & trans)
         {
            Vector3 directionNorm(trans.look_direction.unitVector());
            viz::converters::convert_fzx_to_gl(trans.pos, pos_);
            viz::converters::convert_fzx_to_gl(directionNorm, direction_);
            viz::converters::convert_fzx_to_gl(trans.up, up_);
         }

         // Initialize the camera's state with position, direction, and up
         // vectors in FZX coordinates.
         void initialize(
            const viz::types::cameraTransform_t & trans,
            unsigned int widthIn,
            unsigned int heightIn
         )
         {
            width_ = widthIn;
            height_ = heightIn;

            setTransform(trans);

            projection_ = glm::perspective(
               glm::radians(45.0f),
               (float )width_/(float )height_,
               0.1f,
               far_plane_
            );
         }

         // Sets the camera position in FZX coordinates.
         void setPos(const Vector3 & pos)
         {
            viz::converters::convert_fzx_to_gl(pos, pos_);
         }

         // Sets the direction the camera is looking in FZX coordinates.
         void setLookDirection(const Vector3 & lookDirection)
         {
            Vector3 lookDirectionUnit = lookDirection.unitVector();
            viz::converters::convert_fzx_to_gl(lookDirectionUnit, direction_);
         }

         glm::mat4 & getView(void)
         {
            view_ = glm::lookAt(pos_, pos_ + direction_, up_);
            return view_;
         }

         glm::mat4 & getProjection(void)
         {
            return projection_;
         }

         glm::vec3 & getPos(void)
         {
            return pos_;
         }

         glm::vec3 & getLookDir(void)
         {
            return direction_;
         }

         void setFarPlane(float far_plane)
         {
            far_plane_ = far_plane;
         }

         float getFarPlane(void) const
         {
            return far_plane_;
         }

         // Update the perspective projection based on the window's frame
         // buffer width and height.
         void updateProjection(unsigned int frameBufferWidth, unsigned int frameBufferHeight)
         {
            projection_ = glm::perspective(
               glm::radians(45.0f),
               static_cast<float>(frameBufferWidth)/static_cast<float>(frameBufferHeight),
               0.1f,
               far_plane_
            );
         }

         // Update the camera's understanding of the window's width and height
         // in pixels.
         void updateWindowDimensions(unsigned int width, unsigned int height)
         {
            width_ = width;
            height_ = height;
         }

         // Returns the unit vector direction of the ray cast from the mouse's
         // position on the screen.
         // Implementation courtesy of Anton Gerdelan's blog.
         Vector3 mouseRayCast(double mouseX, double mouseY)
         {
            view_ = glm::lookAt(pos_, pos_ + direction_, up_);

            glm::vec4 gl_ray_clip(
               ((2.f * mouseX) / width_ - 1.f),
               (1.f - (2.f * mouseY) / height_),
               -1.f,
               1.f
            );

            glm::vec4 gl_ray_eye(glm::inverse(projection_) * gl_ray_clip);
            gl_ray_eye.z = -1.f;
            gl_ray_eye.w = 0.f;

            glm::vec4 gl_ray_world4 = (glm::inverse(view_) * gl_ray_eye);
            glm::vec3 gl_ray_world(gl_ray_world4.x, gl_ray_world4.y, gl_ray_world4.z);
            gl_ray_world = glm::normalize(gl_ray_world);

            Vector3 fzx_ray_world;
            viz::converters::convert_gl_to_fzx(gl_ray_world, fzx_ray_world);

            fzx_ray_world[1] *= -1.f;
            fzx_ray_world[2] *= -1.f;

            return fzx_ray_world;
         }

      private:

         // Position of the camera in OpenGL coordinates
         glm::vec3 pos_;

         // Direction the camera is pointing in OpenGL coordinates
         glm::vec3 direction_;

         // The direction that the camera's body-frame up direction is pointing
         // in OpenGL coordinates
         glm::vec3 up_;

         // Width of the camera frame.
         unsigned int width_;

         // Height of the camera frame.
         unsigned int height_;

         glm::mat4 view_;

         glm::mat4 projection_;

         float far_plane_;

   };

}

#endif
