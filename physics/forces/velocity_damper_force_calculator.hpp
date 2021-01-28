#ifndef VELOCITY_DAMPER_CALCULATOR_HEADER
#define VELOCITY_DAMPER_CALCULATOR_HEADER

#include "allocator.hpp"

#include "system.hpp"

namespace oy
{
   class VelocityDamperForceCalculator : public trecs::System
   {
      public:
         void registerComponents(trecs::Allocator & allocator) const override;

         void registerQueries(trecs::Allocator & allocator) override;

         void update(trecs::Allocator & allocator) const;

      private:
         trecs::query_t damper_query_;
   };
}

#endif
