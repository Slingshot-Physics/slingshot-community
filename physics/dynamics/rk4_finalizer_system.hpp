#ifndef RK4_FINALIZE_SYSTEM_HEADER
#define RK4_FINALIZE_SYSTEM_HEADER

#include "allocator.hpp"
#include "system.hpp"

namespace oy
{
   class Rk4Finalizer : public trecs::System
   {
      public:
         void registerComponents(trecs::Allocator & allocator) const override;

         void registerQueries(trecs::Allocator & allocator) override;

         void update(trecs::Allocator & allocator) const;

      private:
         trecs::query_t dynamic_body_query_;
   };
}

#endif
