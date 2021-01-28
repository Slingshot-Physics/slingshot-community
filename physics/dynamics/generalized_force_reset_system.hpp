#ifndef GENERALIZED_FORCE_RESET_SYSTEM_HEADER
#define GENERALIZED_FORCE_RESET_SYSTEM_HEADER

#include "allocator.hpp"

#include "system.hpp"

namespace oy
{

   class GeneralizedForceResetSystem : public trecs::System
   {
      public:
         void registerComponents(trecs::Allocator & allocator) const override;

         void registerQueries(trecs::Allocator & allocator) override;

         void update(trecs::Allocator & allocator) const;

      private:
         trecs::query_t generalized_force_query_;
   };
}

#endif
