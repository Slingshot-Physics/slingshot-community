#include "veggie_state.hpp"

#include "random_utils.hpp"

#include <iostream>

VeggieState::VeggieState(
   const trecs::uid_t & body_id,
   const viz::types::basic_color_t & color,
   const float & point_value,
   unsigned int frames_between_jumps,
   unsigned int frames_per_jump
)
   : body_id(body_id)
   , collected(false)
   , color(color)
   , point_value(point_value)
   , frames_between_jumps(frames_between_jumps)
   , frames_per_jump(frames_per_jump)
{ }

Vector3 VeggieState::update(oy::Handle & handle)
{
   Vector3 jump_force;

   unsigned int count = handle.getFrameCount();
   unsigned int shift_count = count - (count / frames_between_jumps) * frames_between_jumps;

   if (shift_count >= 0 && shift_count <= frames_per_jump && !collected)
   {
      jump_force.Initialize(0.f, 20.f, 50.f);
      jump_force += edbdmath::random_vec3(-15.f, 15.f, -20.f, 40.f, 0.f, 80.f);
   }

   return jump_force;
}
