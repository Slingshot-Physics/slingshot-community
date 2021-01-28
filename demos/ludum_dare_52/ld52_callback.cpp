#include "ld52_callback.hpp"

#include "random_utils.hpp"

#include <chrono>
#include <unordered_set>
#include <thread>
#include <iostream>

void BuggyCallback::post_setup(oy::Handle & handle)
{
   buggy_base_id_ = handle.getIdMapping().at(1);

   std::cout << "buggy base id: " << buggy_base_id_ << "\n";

   fl_wheel_id_ = handle.getIdMapping().at(2);
   fr_wheel_id_ = handle.getIdMapping().at(3);
   bl_wheel_id_ = handle.getIdMapping().at(4);
   br_wheel_id_ = handle.getIdMapping().at(5);

   fl_axle_id_ = handle.getIdMapping().at(6);
   fr_axle_id_ = handle.getIdMapping().at(7);
   bl_axle_id_ = handle.getIdMapping().at(8);
   br_axle_id_ = handle.getIdMapping().at(9);

   fl_hinge_id_ = handle.getIdMapping().at(10);
   fr_hinge_id_ = handle.getIdMapping().at(11);
   bl_hinge_id_ = handle.getIdMapping().at(12);
   br_hinge_id_ = handle.getIdMapping().at(13);

   buggy_base_ = &(handle.getBody(buggy_base_id_));
   fl_wheel_ = &(handle.getBody(fl_wheel_id_));
   fr_wheel_ = &(handle.getBody(fr_wheel_id_));
   bl_wheel_ = &(handle.getBody(bl_wheel_id_));
   br_wheel_ = &(handle.getBody(br_wheel_id_));
   fl_axle_ = &(handle.getBody(fl_axle_id_));
   fr_axle_ = &(handle.getBody(fr_axle_id_));
   bl_axle_ = &(handle.getBody(bl_axle_id_));
   br_axle_ = &(handle.getBody(br_axle_id_));
   fl_hinge_ = &(handle.getBody(fl_hinge_id_));
   fr_hinge_ = &(handle.getBody(fr_hinge_id_));
   bl_hinge_ = &(handle.getBody(bl_hinge_id_));
   br_hinge_ = &(handle.getBody(br_hinge_id_));

   initial_buggy_states_[buggy_base_id_] = *buggy_base_;
   initial_buggy_states_[fl_wheel_id_] = *fl_wheel_;
   initial_buggy_states_[fr_wheel_id_] = *fr_wheel_;
   initial_buggy_states_[bl_wheel_id_] = *bl_wheel_;
   initial_buggy_states_[br_wheel_id_] = *br_wheel_;
   initial_buggy_states_[fl_axle_id_] = *fl_axle_;
   initial_buggy_states_[fr_axle_id_] = *fr_axle_;
   initial_buggy_states_[bl_axle_id_] = *bl_axle_;
   initial_buggy_states_[br_axle_id_] = *br_axle_;
   initial_buggy_states_[fl_hinge_id_] = *fl_hinge_;
   initial_buggy_states_[fr_hinge_id_] = *fr_hinge_;
   initial_buggy_states_[bl_hinge_id_] = *bl_hinge_;
   initial_buggy_states_[br_hinge_id_] = *br_hinge_;

   const auto & revolute_motor_uids = handle.getRevoluteMotorConstraintUids();

   for (const auto revolute_motor_uid : revolute_motor_uids)
   {
      oy::types::constraintRevoluteMotor_t & temp_motor = handle.getRevoluteMotorConstraint(revolute_motor_uid);
      oy::types::bodyLink_t motor_link = handle.getBodyLink(revolute_motor_uid);
      if (
         (motor_link.parentId == bl_wheel_id_ && motor_link.childId == br_wheel_id_) ||
         (motor_link.parentId == br_wheel_id_ && motor_link.childId == bl_wheel_id_)
      )
      {
         drive_motor_ = &temp_motor;
      }

      if (
         (motor_link.parentId == buggy_base_id_ && motor_link.childId == fl_axle_id_) ||
         (motor_link.parentId == fl_axle_id_ && motor_link.childId == buggy_base_id_)
      )
      {
         fl_steer_motor_ = &temp_motor;
      }

      if (
         (motor_link.parentId == buggy_base_id_ && motor_link.childId == fr_axle_id_) ||
         (motor_link.parentId == fr_axle_id_ && motor_link.childId == buggy_base_id_)
      )
      {
         fr_steer_motor_ = &temp_motor;
      }
   }

   if (!veggie_bodies_spawned_)
   {
      spawn_veggies(handle);
      veggie_bodies_spawned_ = true;
   }
}

bool BuggyCallback::operator()(oy::Handle & handle)
{
   if (gui_.getState().show_instructions)
   {
      return false;
   }
   if (gui_.getState().lives_remaining == 0)
   {
      return false;
   }
   if (gui_.getState().win)
   {
      return false;
   }

   jump_veggies(handle);

   if (drive_motor_ != nullptr)
   {
      drive_motor_->angularSpeed = motor_speed_;
   }

   Matrix33 R_W_to_Buggy = buggy_base_->ql2b.rotationMatrix();
   Matrix33 R_W_to_FLA = fl_axle_->ql2b.rotationMatrix();
   Matrix33 R_W_to_FRA = fr_axle_->ql2b.rotationMatrix();

   Matrix33 R_FLA_to_Buggy = R_W_to_Buggy * R_W_to_FLA.transpose();
   Matrix33 R_FRA_to_Buggy = R_W_to_Buggy * R_W_to_FRA.transpose();

   Vector3 x_hat(1.f, 0.f, 0.f);
   Vector3 y_hat(0.f, 1.f, 0.f);

   float fl_axle_error = -1.f * sinf(steer_angle_) - (R_FLA_to_Buggy * y_hat)[0];

   float fr_axle_error = -1.f * sinf(steer_angle_) - (R_FRA_to_Buggy * y_hat)[0];

   if (fl_steer_motor_ != nullptr)
   {
      fl_steer_motor_->angularSpeed = k_steering_ * fl_axle_error;
   }

   if (fr_steer_motor_ != nullptr)
   {
      fr_steer_motor_->angularSpeed = k_steering_ * fr_axle_error;
   }

   count_ += 1;

   if ((count_ % 2) != 0)
   {
      check_veggie_collection(handle);
   }

   update_camera();

   if ((count_ % 10) != 0)
   {
      Vector3 body_ray_unit(0.f, 1.f, -0.3f);
      body_ray_unit.Normalize();
      Vector3 ray_unit(R_W_to_Buggy.transpose() * body_ray_unit);
      Vector3 ray_start = buggy_base_->linPos + 2.f * (R_W_to_Buggy.transpose() * y_hat);
      find_grabbable_entities(handle, ray_start, ray_unit);

      // If the grabber is activated, check if there's an entity in the world
      // that can be grabbed. If disactivated, release the entity.
      if (grabber_pressed_)
      {
         grab_entity(handle);
      }
      else
      {
         release_entity(handle);
      }
   }

   respawn_fallen_veggies(handle);

   if ((buggy_base_->linPos[2] < -64.f) || reset_pressed_)
   {
      respawn_player(handle);
      lives_remaining_ = (lives_remaining_ > 0) ? lives_remaining_ - 1 : 0;
      std::cout << "lives remaining: " << lives_remaining_ << "\n";
      reset_pressed_ = false;
   }

   if (count_ % 100 == 0)
   {
      std::cout << "num sim steps: " << count_ << std::endl;
   }

   // Roll it!
   return true;
}

viz::HIDInterface & BuggyCallback::hid(void)
{
   viz::HIDInterface & hid_controller = camera_controller_;
   return hid_controller;
}

viz::Camera & BuggyCallback::camera(void)
{
   return camera_;
}

void BuggyCallback::update_camera(void)
{
   float yaw_deg, pitch_deg;
   camera_controller_.getYawPitchDeg(yaw_deg, pitch_deg);

   float yaw_rad = yaw_deg * M_PI/180.f;
   float pitch_rad = pitch_deg * M_PI/180.f;

   // Elliptical orbit - farther away if you're looking straight down, closer
   // if you're looking at the car laterally.
   Vector3 orbit_pos_body(
      orbit_radius_ * 2.f * cos(-1.f * pitch_rad) * sin(-1.f * yaw_rad),
      orbit_radius_  * 2.f * -1.f * cos(-1.f * pitch_rad) * cos(-1.f * yaw_rad),
      orbit_radius_ * 3.f * sin(-1.f * pitch_rad)
   );

   Matrix33 R_Buggy_to_W = buggy_base_->ql2b.rotationMatrix().transpose();
   Vector3 orbit_pos_W(R_Buggy_to_W * orbit_pos_body + buggy_base_->linPos);

   Vector3 look_direction(buggy_base_->linPos - orbit_pos_W);
   look_direction.Normalize();

   camera_.setPos(orbit_pos_W);
   camera_.setLookDirection(look_direction);

}

void BuggyCallback::parse_gui(
   viz::VizRenderer * renderer, std::map<trecs::uid_t, int> & fzx_to_viz_ids
)
{
   if (count_ % 10 != 0)
   {
      return;
   }

   LD52GuiState_t & gui_state = gui_.getState();

   gui_state.score = score_;
   gui_state.lives_remaining = lives_remaining_;
   gui_state.win = all_beans_collected_;

   const viz::types::basic_color_t default_color{1.f, 0.647f, 0.f, 1.f};
   const viz::types::basic_color_t grabbable_color{0.f, 0.647f, 1.f, 1.f};
   const viz::types::basic_color_t grabbed_color{1.f, 0.147f, 0.2f, 1.f};

   if (!veggie_meshes_spawned_)
   {
      map_veggie_body_ids_to_mesh_ids(renderer);

      for (const auto veggie_sim_to_viz : veggie_body_to_mesh_ids_)
      {
         fzx_to_viz_ids[veggie_sim_to_viz.first] = veggie_sim_to_viz.second;
      }
      veggie_meshes_spawned_ = true;
   }

   for (const auto sim_to_viz_id : fzx_to_viz_ids)
   {
      if (
         (sim_to_viz_id.first == buggy_base_id_) ||
         (sim_to_viz_id.first == fl_wheel_id_) ||
         (sim_to_viz_id.first == fr_wheel_id_) ||
         (sim_to_viz_id.first == bl_wheel_id_) ||
         (sim_to_viz_id.first == br_wheel_id_) ||
         (sim_to_viz_id.first == fl_axle_id_) ||
         (sim_to_viz_id.first == fr_axle_id_)
      )
      {
         continue;
      }

      if (sim_to_viz_id.first != grabbed_body_id_)
      {
         renderer->updateMeshColor(sim_to_viz_id.second, default_color);
      }
   }

   for (unsigned int i = 0; i < veggie_states_.size(); ++i)
   {
      renderer->updateMeshColor(veggie_body_to_mesh_ids_[veggie_states_[i].body_id], veggie_states_[i].color);
   }

   if (grabbable_entities_.size() > 0)
   {
      renderer->updateMeshColor(fzx_to_viz_ids[grabbable_entities_[0]], grabbable_color);
   }

   if (grabbed_body_id_ != -1)
   {
      renderer->updateMeshColor(fzx_to_viz_ids[grabbed_body_id_], grabbed_color);
   }

   if (!ray_cast_mesh_spawned_)
   {
      viz::types::basic_color_t red = {1.f, 0.f, 0.f, 1.f};
      Vector3 points[2] = {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
      ray_cast_mesh_id_ = renderer->addSegment(2, points, red);
   }

   Matrix33 R_W_to_Buggy = buggy_base_->ql2b.rotationMatrix();
   Vector3 y_hat(0.f, 1.f, 0.f);
   Vector3 body_ray_unit(0.f, 1.f, -0.3f);
   body_ray_unit.Normalize();
   Vector3 ray_unit(R_W_to_Buggy.transpose() * body_ray_unit);
   Vector3 ray_start = buggy_base_->linPos + 1.8f * (R_W_to_Buggy.transpose() * y_hat);
   Vector3 points[2] = {ray_start, ray_start + 15.f * ray_unit};
   renderer->updateSegment(ray_cast_mesh_id_, 2, points);
   ray_cast_mesh_spawned_ = true;
}

void BuggyCallback::find_grabbable_entities(
   oy::Handle & handle, const Vector3 & ray_start, const Vector3 & ray_unit
)
{
   const float max_distance = 15.f;
   grabbable_entities_.clear();

   auto raycast_result = handle.raycast(ray_start, ray_unit, max_distance);

   if (raycast_result.bodyId != -1)
   {
      grabbable_entities_.push_back(raycast_result.bodyId);
   }
}

void BuggyCallback::grab_entity(oy::Handle & handle)
{
   // Only need to add a spring to this entity if one doesn't already exist.
   if ((grabber_spring_id_ == -1) && (grabbable_entities_.size() > 0))
   {
      std::cout << "grabbed entity\n";
      trecs::uid_t grabbable_body_id = grabbable_entities_[0];
      oy::types::rigidBody_t & grabbable_body = handle.getBody(grabbable_body_id);

      Vector3 y_hat(0.f, 1.f, 0.f);
      Vector3 buggy_link_pos = buggy_base_->linPos + 1.8f * buggy_base_->ql2b.rotationMatrix().transpose() * y_hat;

      oy::types::forceSpring_t grabber_spring;
      grabber_spring.parentLinkPoint.Initialize(0.f, 1.8f, 2.f);
      grabber_spring.childLinkPoint.Initialize(0.f, 0.f, 0.f);
      grabber_spring.restLength = 1.f;
      grabber_spring.springCoeff = -10.f;
      grabber_spring_id_ = handle.addSpringForce(buggy_base_id_, grabbable_entities_[0], grabber_spring);

      grabbed_body_id_ = grabbable_entities_[0];
   }
}

void BuggyCallback::release_entity(oy::Handle & handle)
{
   if (grabber_spring_id_ != -1)
   {
      std::cout << "released entity\n";
      handle.removeEntity(grabber_spring_id_);
      grabber_spring_id_ = -1;
      grabbed_body_id_ = -1;
   }
}

void BuggyCallback::spawn_veggies(oy::Handle & handle)
{
   const float radius = 0.18f * 1.902113047;
   const float height = 0.18f * 2.f;
   for (int i = 0; i < max_num_veggies_; ++i)
   {
      oy::types::rigidBody_t temp_body;
      temp_body.linPos = edbdmath::random_vec3(-18.f, 18.f, -18.f, 18.f, 13.f, 15.f);
      temp_body.angVel = edbdmath::random_vec3(-1.f, 1.f, -2.f, 2.f, -3.f, 3.f);
      temp_body.mass = 0.5f;
      temp_body.inertiaTensor = identityMatrix();
      temp_body.inertiaTensor(0, 0) = (1.f / 12.f) * temp_body.mass * (
         3.f * radius * radius + height * height
      );
      temp_body.inertiaTensor(1, 1) = (0.5f) * temp_body.mass * (radius * radius);
      temp_body.inertiaTensor(2, 2) = (1.f / 12.f) * temp_body.mass * (
         3.f * radius * radius + height * height
      );
      temp_body.ql2b.Initialize(1.f, 0.f, 0.f, 0.f);

      oy::types::isometricCollider_t temp_collider;
      temp_collider.enabled = true;
      temp_collider.mu = 0.5f;
      temp_collider.restitution = 0.9f;
      // temp_collider.scale = 0.18f * identityMatrix();

      // Load a capsule shape with a radius and height that's the same as the
      // default radius and height of the capsule's triangle mesh.
      geometry::types::shape_t temp_shape;
      temp_shape.shapeType = geometry::types::enumShape_t::CAPSULE;
      temp_shape.capsule.radius = radius;
      temp_shape.capsule.height = height;

      trecs::uid_t new_body_id = handle.addBody(
         temp_body, temp_collider, temp_shape, oy::types::enumRigidBody_t::DYNAMIC
      );
      if (new_body_id == -1)
      {
         std::cout << "couldn't add a veggie?\n";
         return;
      }

      oy::types::forceConstant_t gravity;
      gravity.acceleration.Initialize(0.f, 0.f, -9.8f);
      gravity.forceFrame = oy::types::enumFrame_t::GLOBAL;
      gravity.childLinkPoint.Initialize(0.f, 0.f, 0.f);
      handle.addConstantForce(new_body_id, gravity);

      float rando_numbo = edbdmath::random_float();

      viz::types::basic_color_t aubergine{0.28235f, 0.047058f, 0.41176f, 1.f};
      viz::types::basic_color_t squash{0.949019f, 0.670588f, 0.082352f, 1.f};
      viz::types::basic_color_t pumpkin{1.0f, 0.459f, 0.094f, 1.f};
      viz::types::basic_color_t green_pepper{0.3294f, 0.4353f, 0.1333f, 1.f};
      viz::types::basic_color_t tomato{1.f, 0.3882f, 0.2784f, 1.f};

      viz::types::basic_color_t veggie_color;
      if (rando_numbo < 0.25f)
      {
         veggie_color = squash;
      }
      else if (rando_numbo < 0.5f)
      {
         veggie_color = pumpkin;
      }
      else if (rando_numbo < 0.75f)
      {
         veggie_color = green_pepper;
      }
      else
      {
         veggie_color = tomato;
      }

      if (i < 3)
      {
         veggie_color = aubergine;
      }

      VeggieState temp_veggie_state(
         new_body_id,
         veggie_color,
         (i < 3) ? 50.f : 5.f,
         (unsigned int )edbdmath::random_float(2500, 5000),
         (unsigned int )edbdmath::random_float(10, 20)
      );
      veggie_states_.push_back(temp_veggie_state);
   }
}

void BuggyCallback::map_veggie_body_ids_to_mesh_ids(viz::VizRenderer * renderer)
{
   veggie_body_to_mesh_ids_.clear();

   data_triangleMesh_t veggie_mesh_data = \
      geometry::mesh::loadDefaultShapeMeshData(
         geometry::types::enumShape_t::CAPSULE, 0.18f
      );

   for (const auto veggie_state : veggie_states_)
   {
      int veggie_mesh_id = renderer->addMesh(veggie_mesh_data, 0);
      veggie_body_to_mesh_ids_[veggie_state.body_id] = veggie_mesh_id;
   }
}

void BuggyCallback::respawn_fallen_veggies(oy::Handle & handle)
{
   for (unsigned int i = 0; i < veggie_states_.size(); ++i)
   {
      oy::types::rigidBody_t & veggie_state = handle.getBody(veggie_states_[i].body_id);

      if (veggie_state.linPos[2] < -64.f)
      {
         veggie_state.linPos = edbdmath::random_vec3(-28.f, 28.f, -28.f, 28.f, 10.f, 15.f);
         veggie_state.linVel.Initialize(0.f, 0.f, 0.f);
      }
   }
}

void BuggyCallback::respawn_player(oy::Handle & handle)
{
   for (const auto sim_id_to_state : initial_buggy_states_)
   {
      handle.getBody(sim_id_to_state.first) = sim_id_to_state.second;
   }
   release_entity(handle);
}

void BuggyCallback::check_veggie_collection(oy::Handle & handle)
{
   bool all_veggies_collected = true;
   for (unsigned int i = 0; i < veggie_states_.size(); ++i)
   {
      if (veggie_states_[i].collected)
      {
         continue;
      }

      all_veggies_collected = false;

      geometry::types::aabb_t veggie_aabb = handle.getAabb(veggie_states_[i].body_id);

      bool bodyAAheadBodyB = veggie_aabb.vertMin[0] > bucket_box_.vertMax[0];
      bool bodyABehindBodyB = veggie_aabb.vertMax[0] < bucket_box_.vertMin[0];

      bool bodyALeftBodyB = veggie_aabb.vertMin[1] > bucket_box_.vertMax[1];
      bool bodyARightBodyB = veggie_aabb.vertMax[1] < bucket_box_.vertMin[1];

      bool bodyAAboveBodyB = veggie_aabb.vertMin[2] > bucket_box_.vertMax[2];
      bool bodyABelowBodyB = veggie_aabb.vertMax[2] < bucket_box_.vertMin[2];

      if (
         !(
            bodyAAheadBodyB ||
            bodyABehindBodyB ||
            bodyAAboveBodyB ||
            bodyABelowBodyB ||
            bodyALeftBodyB ||
            bodyARightBodyB
         )
      )
      {
         std::cout << "veggie id: " << veggie_states_[i].body_id << " hit the bucket box!\n";
         score_ += veggie_states_[i].point_value;
         std::cout << "current score: " << score_ << "\n";
         veggie_states_[i].collected = true;
      }
   }

   all_beans_collected_ |= all_veggies_collected;
}

void BuggyCallback::jump_veggies(oy::Handle & handle)
{
   for (unsigned int i = 0; i < 3; ++i)
   {
      handle.applyForce(
         veggie_states_[i].body_id,
         veggie_states_[i].update(handle),
         oy::types::enumFrame_t::GLOBAL
      );
   }
}
