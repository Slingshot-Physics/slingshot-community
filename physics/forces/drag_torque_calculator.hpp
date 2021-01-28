#ifndef DRAG_TORQUE_CALCULATOR_SYSTEM
#define DRAG_TORQUE_CALCULATOR_SYSTEM

#include "allocator.hpp"

#include "system.hpp"

namespace oy
{
   class DragTorqueCalculator : public trecs::System
   {
      public:
         void registerComponents(trecs::Allocator & allocator) const override;

         void registerQueries(trecs::Allocator & allocator) override;

         void update(trecs::Allocator & allocator) const;

      private:
         trecs::query_t drag_query_;
   };
}

#endif
