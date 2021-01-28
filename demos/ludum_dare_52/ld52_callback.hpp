#ifndef LD52_CALLBACK_HEADER
#define LD52_CALLBACK_HEADER

#include "slingshot_callback_base.hpp"

#include "ld52_camera_controller.hpp"
#include "ld52_gui.hpp"

#include "veggie_state.hpp"

#include <deque>
#include <map>
#include <vector>

// Applies a force over the bump duration to the base of the cart.
class BuggyCallback : public slingshot::CallbackBase
{
   public:
      BuggyCallback(void)
         : count_(0)
         , lives_remaining_(5)
         , score_(0.f)
         , all_beans_collected_(false)
         , gui_(lives_remaining_)
         , bucket_box_{
            {3.f, 3.f, 8.1f}, {-3.f, -3.f, 7.1f}, 
         }
         , camera_offset_(0.f, -5.f, 4.f)
         , camera_controller_(
            camera_,
            motor_speed_,
            steer_angle_,
            grabber_pressed_,
            reset_pressed_
         )
         , ray_cast_mesh_id_(-1)
         , ray_cast_mesh_spawned_(false)
         , reset_pressed_(false)
         , grabber_pressed_(false)
         , grabber_spring_id_(-1)
         , grabbed_body_id_(-1)
         , motor_speed_(0.f)
         , k_steering_(5.f)
         , steer_angle_(0.f)
         , orbit_radius_(5.f)
         , drive_motor_(nullptr)
         , fl_steer_motor_(nullptr)
         , fr_steer_motor_(nullptr)
         , buggy_base_(nullptr)
         , fl_wheel_(nullptr)
         , fr_wheel_(nullptr)
         , bl_wheel_(nullptr)
         , br_wheel_(nullptr)
         , fl_axle_(nullptr)
         , fr_axle_(nullptr)
         , bl_axle_(nullptr)
         , br_axle_(nullptr)
         , fl_hinge_(nullptr)
         , fr_hinge_(nullptr)
         , bl_hinge_(nullptr)
         , br_hinge_(nullptr)
         , veggie_bodies_spawned_(false)
         , veggie_meshes_spawned_(false)
         , max_num_veggies_(15)
      { }

      void post_setup(oy::Handle & handle);

      bool operator()(oy::Handle & handle);

      viz::HIDInterface & hid(void);

      viz::Camera & camera(void);

      void update_camera(void);

      void parse_gui(
         viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
      );

      viz::GuiCallbackBase * gui(void)
      {
         return &gui_;
      }

   private:
      typedef std::deque<trecs::uid_t>::iterator dui_it_t;

      // Number of sim updates
      unsigned int count_;

      unsigned int lives_remaining_;

      float score_;

      bool all_beans_collected_;

      LD52Gui gui_;

      geometry::types::aabb_t bucket_box_;

      Vector3 camera_offset_;

      viz::Camera camera_;

      viz::BuggyCameraController camera_controller_;

      int ray_cast_mesh_id_;

      bool ray_cast_mesh_spawned_;

      bool reset_pressed_;

      bool grabber_pressed_;

      trecs::uid_t grabber_spring_id_;

      trecs::uid_t grabbed_body_id_;

      float motor_speed_;

      float k_steering_;

      float steer_angle_;

      float orbit_radius_;

      trecs::uid_t buggy_base_id_;

      trecs::uid_t fl_wheel_id_;

      trecs::uid_t fr_wheel_id_;

      trecs::uid_t bl_wheel_id_;

      trecs::uid_t br_wheel_id_;

      trecs::uid_t fl_axle_id_;

      trecs::uid_t fr_axle_id_;

      trecs::uid_t bl_axle_id_;

      trecs::uid_t br_axle_id_;

      trecs::uid_t fl_hinge_id_;

      trecs::uid_t fr_hinge_id_;

      trecs::uid_t bl_hinge_id_;

      trecs::uid_t br_hinge_id_;

      oy::types::constraintRevoluteMotor_t * drive_motor_;

      oy::types::constraintRevoluteMotor_t * fl_steer_motor_;

      oy::types::constraintRevoluteMotor_t * fr_steer_motor_;

      oy::types::rigidBody_t * buggy_base_;

      oy::types::rigidBody_t * fl_wheel_;

      oy::types::rigidBody_t * fr_wheel_;

      oy::types::rigidBody_t * bl_wheel_;

      oy::types::rigidBody_t * br_wheel_;

      oy::types::rigidBody_t * fl_axle_;

      oy::types::rigidBody_t * fr_axle_;

      oy::types::rigidBody_t * bl_axle_;

      oy::types::rigidBody_t * br_axle_;

      oy::types::rigidBody_t * fl_hinge_;

      oy::types::rigidBody_t * fr_hinge_;

      oy::types::rigidBody_t * bl_hinge_;

      oy::types::rigidBody_t * br_hinge_;

      std::deque<trecs::uid_t> grabbable_entities_;

      bool veggie_bodies_spawned_;

      bool veggie_meshes_spawned_;

      const int max_num_veggies_;

      std::vector<VeggieState> veggie_states_;

      std::map<trecs::uid_t, int> veggie_body_to_mesh_ids_;

      std::map<trecs::uid_t, oy::types::rigidBody_t> initial_buggy_states_;

      void find_grabbable_entities(
         oy::Handle & handle, const Vector3 & ray_start, const Vector3 & ray_unit
      );

      void grab_entity(oy::Handle & handle);

      void release_entity(oy::Handle & handle);

      // Should only be called once.
      void spawn_veggies(oy::Handle & handle);

      // Should only be called once.
      void map_veggie_body_ids_to_mesh_ids(viz::VizRenderer * renderer);

      // Used to respawn veggies that have fallen off of the map.
      void respawn_fallen_veggies(oy::Handle & handle);

      // Puts the player in the exact same position they started in.
      void respawn_player(oy::Handle & handle);

      void check_veggie_collection(oy::Handle & handle);

      void jump_veggies(oy::Handle & handle);
};

#endif
