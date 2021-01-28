#ifndef CONSTRAINT_STOVE_HEADER
#define CONSTRAINT_STOVE_HEADER

#include "dynamic_array.hpp"
#include "slingshot_types.hpp"
#include "graph.hpp"
#include "heap_matrix.hpp"
#include "subgraph.hpp"

#include <bitset>
#include <cmath>
#include <unordered_map>

namespace oy
{

   class ConstraintStove
   {
      struct lifetime_val_t
      {
         float val;
         unsigned int count;
      };

      struct running_average_t
      {
         float count;
         float average;

         running_average_t(void)
            : count(0.f)
            , average(0.f)
         { }

         running_average_t(float start_val)
            : count(1.f)
            , average(start_val)
         { }

         void update(float new_val)
         {
            float sum = average * count + new_val;
            count += 1.f;
            average = sum / count;
         }
      };

      public:

         ConstraintStove(unsigned int max_lifetime);

         // Push new values onto the existing entries, or add new entries.
         void push(
            const HeapMatrix & lambda_dt,
            const graph::Subgraph<trecs::uid_t, oy::types::edgeId_t> & subgraph
         );

         // Populates the given lamdba_dt using any values on the stove.
         void pull(
            HeapMatrix & lambda_dt,
            const graph::Subgraph<trecs::uid_t, oy::types::edgeId_t> & subgraph,
            const float default_val
         ) const;

         // Increment the lifetime counts of all of the values in the stove.
         // Any values in the stove exceeding the maximum lifetime are pruned.
         void increment(void);

         // The key for a constraint is 96 bits. The upper 32 bits are the
         // constraint ID. The middle 32 bits are body A's ID. The lowest 32
         // bits are body B's ID.
         inline std::bitset<96> calculateStandardKey(
            const graph::types::labeled_edge_t<trecs::uid_t, oy::types::edgeId_t> edge
         ) const
         {
            std::bitset<96> hash;
            hash = static_cast<int>(edge.edgeId.constraintType);
            hash <<= 32;
            hash |= static_cast<int>(edge.nodeIdA);
            hash <<= 32;
            hash |= static_cast<int>(edge.nodeIdB);

            return hash;
         }

         unsigned int size(void) const
         {
            return standard_stove_.size();
         }

      private:

         unsigned int max_lifetime_;

         std::unordered_map<std::bitset<96>, lifetime_val_t> standard_stove_;

         // Gets rid of all of the entries on the stove whose number of counts
         // exceeds the maximum lifetime.
         void prune(void);

   };

}

#endif
