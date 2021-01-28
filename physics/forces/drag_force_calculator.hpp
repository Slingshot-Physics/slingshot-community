#ifndef DRAG_FORCE_CALCULATOR_SYSTEM_HEADER
#define DRAG_FORCE_CALCULATOR_SYSTEM_HEADER

#include "allocator.hpp"
#include "system.hpp"

#include "slingshot_types.hpp"

namespace oy
{
   class DragForceCalculator : public trecs::System
   {
      public:
         void registerComponents(trecs::Allocator & allocator) const override;

         void registerQueries(trecs::Allocator & allocator) override;

         void update(trecs::Allocator & allocator) const;

      private:
         trecs::query_t drag_force_query_;
   };
}

#endif
