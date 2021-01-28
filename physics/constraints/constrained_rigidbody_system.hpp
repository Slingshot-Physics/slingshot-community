#ifndef CONSTRAINED_RIGID_BODY_SYSTEM_HEADER
#define CONSTRAINED_RIGID_BODY_SYSTEM_HEADER

#include "allocator.hpp"

#include "system.hpp"

namespace oy
{
   class ConstrainedRigidBodySystem : public trecs::System
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
