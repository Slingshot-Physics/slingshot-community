#ifndef RIGID_BODY_WIDGET_HEADER
#define RIGID_BODY_WIDGET_HEADER

#include "component_widget_interface.hpp"

#include <map>

void shapeComboBox(
   const std::map<geometry::types::enumShape_t, std::string> & shape_type_to_names,
   geometry::types::enumShape_t & current_shape,
   const char * combo_box_label
);

bool shapeUi(geometry::types::shape_t & shape);

class RigidBodyWidget : public IComponentWidget
{

   struct body_viz_id_t
   {
      bool render;
      int meshId;
   };

   struct rpy_deg_t
   {
      Vector3 rpyDeg;
   };

   public:
      RigidBodyWidget(
         trecs::Allocator & allocator, viz::VizRenderer & renderer
      )
         : IComponentWidget(allocator, renderer)
      {
         allocator_.registerComponent<oy::types::StationaryBody>();
         allocator_.registerComponent<oy::types::DynamicBody>();

         allocator_.registerComponent<oy::types::rigidBody_t>();
         allocator_.registerComponent<oy::types::isometricCollider_t>();
         allocator_.registerComponent<geometry::types::shape_t>();
         allocator_.registerComponent<geometry::types::aabb_t>();
         allocator_.registerComponent<rpy_deg_t>();

         allocator_.registerComponent<body_viz_id_t>();

         allocator_.registerComponent<viz::color_t>();

         rigid_body_query_ = allocator_.addArchetypeQuery<oy::types::rigidBody_t>();
      }

      ~RigidBodyWidget(void) override
      { }

      trecs::uid_t addDefaultComponent(void) override;

      void deleteComponent(trecs::uid_t entity) override;

      void componentsUi(void) override;

      trecs::uid_t addRigidBody(
         const oy::types::rigidBody_t & body,
         const oy::types::isometricCollider_t & collider,
         const geometry::types::shape_t & shape,
         const bool stationary
      );

   private:
      trecs::query_t rigid_body_query_;

      const std::map<geometry::types::enumShape_t, std::string> shape_type_to_name_ = {
         {geometry::types::enumShape_t::CUBE, "cube"},
         {geometry::types::enumShape_t::CYLINDER, "cylinder"},
         {geometry::types::enumShape_t::SPHERE, "sphere"},
         {geometry::types::enumShape_t::CAPSULE, "capsule"},
      };

      void rigidBodyUi(
         trecs::uid_t body_entity,
         oy::types::rigidBody_t & body,
         oy::types::isometricCollider_t & collider,
         geometry::types::shape_t & shape,
         rpy_deg_t & rpy_deg,
         body_viz_id_t & viz_id,
         viz::color_t & color,
         const std::string & label
      );

};

#endif
