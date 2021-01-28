#ifndef SPRING_FORCE_CALCULATOR_HEADER
#define SPRING_FORCE_CALCULATOR_HEADER

#include "allocator.hpp"

#include "system.hpp"

namespace oy
{
   class SpringForceCalculator : public trecs::System
   {
      public:
         void registerComponents(trecs::Allocator & allocator) const override;

         void registerQueries(trecs::Allocator & allocator) override;

         void update(trecs::Allocator & allocator) const;

      private:
         trecs::query_t spring_query_;
   };
}

#endif
