#ifndef COMPONENT_WIDGET_INTERFACE_HEADER
#define COMPONENT_WIDGET_INTERFACE_HEADER

#include "allocator.hpp"
#include "viz_renderer.hpp"

#include "slingshot_types.hpp"

// An interface for adding, modifying, and deleting components of a given type.
class IComponentWidget
{
   public:
      IComponentWidget(
         trecs::Allocator & allocator,
         viz::VizRenderer & renderer
      )
         : allocator_(allocator)
         , renderer_(renderer)
      {
         allocator_.registerComponent<oy::types::rigidBody_t>();

         rigid_body_query_ = allocator.addArchetypeQuery<oy::types::rigidBody_t>();
      }

      IComponentWidget(const IComponentWidget &) = delete;

      IComponentWidget & operator=(const IComponentWidget &) = delete;

      virtual ~IComponentWidget(void)
      { }

      virtual trecs::uid_t addDefaultComponent(void) = 0;

      virtual void deleteComponent(trecs::uid_t entity) = 0;

      virtual void componentsUi(void) = 0;

   protected:
      trecs::Allocator & allocator_;

      viz::VizRenderer & renderer_;

      trecs::query_t rigid_body_query_;

      const viz::color_t cyan = {41.f/255.f, 221.f/255.f, 244.f/255.f, 1.f};

      const viz::color_t lavender = {133.f/255.f, 75.f/255.f, 221.f/255.f, 1.f};

      const viz::color_t red = {1.f, 0.f, 0.f, 1.f};

      const viz::color_t green = {0.f, 1.f, 0.f, 1.f};

      const viz::color_t blue = {0.f, 0.f, 1.f, 1.f};

      const viz::color_t gray = {0.5f, 0.5f, 0.5f, 1.f};

      // Builds a combo box out of all possible body UIDs in `body_ids`, except
      // for `excluded_body_id`. Colors the elements of the body ID combo box
      // based on the body colors in the fzx_colors_ map.
      void bodyIdComboBox(
         const std::string & combo_box_label,
         const trecs::uid_t excluded_body_uid,
         const std::unordered_set<trecs::uid_t> & rigid_body_entities,
         trecs::uid_t & current_body_uid
      ) const;

      geometry::types::isometricTransform_t getTransform(
         trecs::uid_t body_entity
      );

};

#endif
