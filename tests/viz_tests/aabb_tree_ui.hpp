#ifndef AABB_TREE_UI_HEADER
#define AABB_TREE_UI_HEADER

#include "allocator.hpp"

#include "aabb_tree.hpp"
#include "slingshot_types.hpp"
#include "geometry_types.hpp"
#include "gui_callback_base.hpp"

#include "viz_renderer.hpp"

#include <vector>

class AabbTreeUi : public viz::GuiCallbackBase
{
   struct aabb_viz_t
   {
      bool enable;
      int id;
   };

   public:
      AabbTreeUi(
         const char * prefix, viz::VizRenderer & renderer
      )
         : prefix_(prefix)
         , allocator_(65536)
         , tree_({-1, {{1.f, 1.f, 1.f}, {-1.f, -1.f, -1.f}}})
         , renderer_(renderer)
         , show_grid_(true)
      {
         allocator_.registerComponent<geometry::types::aabb_t>();

         allocator_.registerComponent<aabb_viz_t>();

         aabb_query_ = allocator_.addArchetypeQuery<geometry::types::aabb_t>();

         renderer_.enableGrid();
      }

      void operator()(void);

      bool no_window(void);

   private:
      std::string prefix_;

      trecs::Allocator allocator_;

      trecs::uid_t aabb_query_;

      geometry::AabbTree tree_;

      viz::VizRenderer & renderer_;

      bool show_grid_;

      std::vector<aabb_viz_t> tree_viz_ids_;

      void update_viz_mesh(
         const int viz_id, const geometry::types::aabb_t & aabb
      );
};

#endif
