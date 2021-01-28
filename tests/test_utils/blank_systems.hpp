#include "system.hpp"


// A basic system implementation.
class SandoSystem : public trecs::System
{
   public:
      virtual ~SandoSystem(void)
      { }

      void registerComponents(trecs::Allocator & allocator) const
      { }

      void registerQueries(trecs::Allocator & allocator) const
      { }
};

// A basic system implementation with an unused const member.
class NandoSystem : public trecs::System
{
   public:
      NandoSystem(void)
         : thing_(12)
      { }

      void registerComponents(trecs::Allocator & allocator) const
      { }

      void registerQueries(trecs::Allocator & allocator) const
      { }

      virtual ~NandoSystem(void)
      { }

   private:
      const int thing_;
};
