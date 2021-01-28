#ifndef VEGGIE_STATE_HEADER
#define VEGGIE_STATE_HEADER

#include "vector3.hpp"

#include "handle.hpp"

#include "viz_types.hpp"

struct VeggieState
{

   VeggieState(
      // const unsigned int & body_id,
      const trecs::uid_t & body_id,
      const viz::types::basic_color_t & color,
      const float & point_value,
      unsigned int frames_between_jumps,
      unsigned int frames_per_jump
   );

   // unsigned int body_id;
   trecs::uid_t body_id;

   bool collected;

   viz::types::basic_color_t color;

   float point_value;

   unsigned int frames_between_jumps;

   unsigned int frames_per_jump;

   Vector3 update(oy::Handle & handle);

};

#endif
