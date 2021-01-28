#ifndef RK4_INTEGRATOR_HEADER
#define RK4_INTEGRATOR_HEADER

#include "allocator.hpp"

#include "system.hpp"

namespace oy
{
   class Rk4Integrator : public trecs::System
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
