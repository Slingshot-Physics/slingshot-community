#ifndef VIZ_TYPES_HEADER
#define VIZ_TYPES_HEADER

#include "vector3.hpp"

#include <map>

namespace viz
{

namespace types
{

   namespace draw_modes
   {
      // Draw modes - for use with `glDraw*` calls.
      typedef enum
      {
         VIZ_TRIANGLES,
         VIZ_LINES,
         VIZ_LINE_LOOP,
      } enum_t;
   }

   namespace polygon_modes
   {
      // Polygon modes - for use with `glPolygonMode`.
      typedef enum
      {
         VIZ_FILL = 0,
         VIZ_LINE = 1,
         VIZ_POINT = 2,
      } enum_t;
   }

   struct vec4_t
   {
      float & operator[](unsigned int i)
      {
         return c[i];
      }

      float operator[](unsigned int i) const
      {
         return c[i];
      }

      float c[4];
   };

   typedef vec4_t basic_color_t;

   struct basic_vertex_t
   {
      float pos[3];
      float color[4];
      float texture[2];
      float textureId;
      float normal[3];
   };

   struct meshProperties_t
   {
      meshProperties_t(void)
         : color{{0.f, 0.f, 0.f, 0.f}}
      { }

      basic_color_t color;
   };

   struct config_t
   {
      config_t(void)
         : cameraPoint{0.f, 0.f, 2.f}
         , cameraPos{0.f, -1.f, 2.f}
      { }

      config_t(
         unsigned int windowWidth,
         unsigned int windowHeight,
         unsigned int maxFps
      )
         : windowWidth(windowWidth)
         , windowHeight(windowHeight)
         , maxFps(maxFps)
         , cameraPoint{0.f, 0.f, 2.f}
         , cameraPos{0.f, -1.f, 2.f}
      { }

      unsigned int windowWidth;
      unsigned int windowHeight;
      unsigned int maxFps;

      Vector3 cameraPoint;
      Vector3 cameraPos;
   };

   struct cameraTransform_t
   {
      Vector3 pos;
      Vector3 up;
      Vector3 look_direction;
   };

}

}

#endif
