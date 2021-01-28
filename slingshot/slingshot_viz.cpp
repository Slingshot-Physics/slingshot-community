#include "slingshot.hpp"

#include "default_callback.hpp"
#include "geometry_type_converters.hpp"
#include "transform.hpp"
#include "viz_renderer.hpp"

#include <iostream>
#include <limits>

namespace slingshot
{
   api::api(
      const data_scenario_t & scenario,
      const data_vizConfig_t & viz_config_data,
      CallbackBase * user_callback
   )
      : handle_(scenario)
      , callback_(user_callback)
      , renderer_(nullptr)
      , use_default_callback_(user_callback == nullptr)
      , frame_start_time_(std::chrono::high_resolution_clock::now())
      , sim_start_time_(std::chrono::high_resolution_clock::now())
      , stop_sim_(false)
      , realtime_(false)
   {
      copy_vizConfig(&viz_config_data, &viz_config_);

      renderer_ = new viz::VizRenderer(viz_config_);

      if (use_default_callback_)
      {
         callback_ = new DefaultCallback();
      }

      callback_->post_setup(handle_);
      callback_->hid().initialize(viz_config_);
      renderer_->setUserPointer(callback_->hid());

      scen_id_to_sim_id_ = handle_.getIdMapping();
      initialize_renderer();
   }

   api::api(
      const data_scenario_t & scenario,
      const data_vizConfig_t & viz_config_data,
      const std::string & window_name,
      CallbackBase * user_callback
   )
      : handle_(scenario)
      , callback_(user_callback)
      , renderer_(nullptr)
      , use_default_callback_(user_callback == nullptr)
      , frame_start_time_(std::chrono::high_resolution_clock::now())
      , sim_start_time_(std::chrono::high_resolution_clock::now())
      , stop_sim_(false)
      , realtime_(false)
   {
      copy_vizConfig(&viz_config_data, &viz_config_);

      renderer_ = new viz::VizRenderer(viz_config_);

      if (use_default_callback_)
      {
         callback_ = new DefaultCallback();
      }

      callback_->post_setup(handle_);
      callback_->hid().initialize(viz_config_);
      renderer_->setUserPointer(callback_->hid());

      scen_id_to_sim_id_ = handle_.getIdMapping();
      initialize_renderer();

      if (renderer_ != nullptr)
      {
         renderer_->setWindowName(window_name);
      }
   }

   api::api(
      const data_vizScenarioConfig_t & viz_scenario_config,
      CallbackBase * user_callback
   )
      : handle_(viz_scenario_config.scenario)
      , callback_(user_callback)
      , renderer_(nullptr)
      , use_default_callback_(user_callback == nullptr)
      , frame_start_time_(std::chrono::high_resolution_clock::now())
      , sim_start_time_(std::chrono::high_resolution_clock::now())
      , stop_sim_(false)
      , realtime_(false)
   {
      copy_vizConfig(&viz_scenario_config.vizConfig, &viz_config_);

      renderer_ = new viz::VizRenderer(viz_config_);

      if (use_default_callback_)
      {
         callback_ = new DefaultCallback();
      }

      callback_->post_setup(handle_);
      callback_->hid().initialize(viz_config_);
      renderer_->setUserPointer(callback_->hid());

      scen_id_to_sim_id_ = handle_.getIdMapping();
      initialize_renderer();
   }

   api::api(
      const data_vizScenarioConfig_t & viz_scenario_config,
      const std::string & window_name,
      CallbackBase * user_callback
   )
      : handle_(viz_scenario_config.scenario)
      , callback_(user_callback)
      , renderer_(nullptr)
      , use_default_callback_(user_callback == nullptr)
      , frame_start_time_(std::chrono::high_resolution_clock::now())
      , sim_start_time_(std::chrono::high_resolution_clock::now())
      , stop_sim_(false)
      , realtime_(false)
   {
      copy_vizConfig(&viz_scenario_config.vizConfig, &viz_config_);

      renderer_ = new viz::VizRenderer(viz_config_);

      if (use_default_callback_)
      {
         callback_ = new DefaultCallback();
      }

      callback_->post_setup(handle_);
      callback_->hid().initialize(viz_config_);
      renderer_->setUserPointer(callback_->hid());

      scen_id_to_sim_id_ = handle_.getIdMapping();
      initialize_renderer();

      if (renderer_ != nullptr)
      {
         renderer_->setWindowName(window_name);
      }
   }

   void api::initialize_renderer(void)
   {
      if (renderer_ == nullptr)
      {
         return;
      }

      for (
         auto id_it = scen_id_to_sim_id_.begin();
         id_it != scen_id_to_sim_id_.end();
         ++id_it
      )
      {
         trecs::uid_t sim_body_id = id_it->second;
         if (sim_body_id == -1)
         {
            continue;
         }

         geometry::types::shape_t shape = handle_.getShape(sim_body_id);
         geometry::types::triangleMesh_t temp_mesh = geometry::mesh::loadShapeMesh(
            shape
         );

         data_triangleMesh_t temp_mesh_data;
         geometry::converters::to_pod(temp_mesh, &temp_mesh_data);

         int renderer_mesh_id = renderer_->addMesh(temp_mesh_data, 0);
         fzx_to_viz_ids_[sim_body_id] = renderer_mesh_id;
      }

      // Assign mesh properties from viz config to the existing meshes.
      for (unsigned int i = 0; i < viz_config_.numMeshProps; ++i)
      {
         data_vizMeshProperties_t & meshProps = viz_config_.meshProps[i];
         int scenario_mesh_id = meshProps.bodyId;

         if (scen_id_to_sim_id_.find(scenario_mesh_id) == scen_id_to_sim_id_.end())
         {
            continue;
         }

         int fzx_mesh_id = scen_id_to_sim_id_[scenario_mesh_id];
         int viz_mesh_id = fzx_to_viz_ids_[fzx_mesh_id];

         renderer_->updateMeshColor(viz_mesh_id, meshProps.color);
      }
   }

   void api::update_renderer(void)
   {
      if (renderer_ == nullptr)
      {
         return;
      }

      update_meshes();
      callback_->parse_gui(renderer_, fzx_to_viz_ids_);
      stop_sim_ = renderer_->draw(callback_->camera(), callback_->gui());
   }

   void api::update_meshes(void)
   {
      // Update viz meshes according to the transforms in the fzx library.
      for (
         auto id_it = fzx_to_viz_ids_.begin();
         id_it != fzx_to_viz_ids_.end();
         ++id_it
      )
      {
         const oy::types::rigidBody_t & body = handle_.getBody(id_it->first);

         Quaternion q_b2l = ~body.ql2b;
         Matrix33 R_b2l = q_b2l.rotationMatrix();

         geometry::types::transform_t trans_C_to_W;
         trans_C_to_W.translate = body.linPos;
         trans_C_to_W.rotate = R_b2l;
         trans_C_to_W.scale = identityMatrix();

         renderer_->updateMeshTransform(
            id_it->second,
            trans_C_to_W.translate,
            trans_C_to_W.rotate,
            trans_C_to_W.scale
         );
      }
   }
}
