#ifndef CONSTRAINT_SOLVER_HEADER
#define CONSTRAINT_SOLVER_HEADER

#include "allocator.hpp"

#include "constraint_stove.hpp"
#include "dynamic_array.hpp"
#include "slingshot_types.hpp"
#include "graph_types.hpp"
#include "system.hpp"

#include <unordered_map>
#include <vector>

namespace oy
{
   // Pulls in the constraint output from inequality and equality constraints,
   // splits up the constraint factor graph into connected components, and
   // solves the constraint impulses for each connected component using
   // Projected Gauss-Seidel.
   class ConstraintSolver : public trecs::System
   {
      public:
         ConstraintSolver(void);

         void registerComponents(trecs::Allocator & allocator) const override;

         void registerQueries(trecs::Allocator & allocator) override;

         void update(trecs::Allocator & allocator);

      private:

         typedef std::vector<oy::types::constraintOutput_t> ineqConstraintOutput_t;

         typedef oy::types::edgeId_t edgeId_t;

         oy::ConstraintStove stove_;

         // Query for edges tied to equality constraints
         trecs::query_t eq_constraint_edge_query_;

         // Query for equality constraint output
         trecs::query_t eq_constraint_query_;

         // Query for edges tied to inequality constraints
         trecs::query_t ineq_constraint_edge_query_;

         // Query for inequality constraint output
         trecs::query_t ineq_constraint_query_;

         // Query for constrained rigid body output
         trecs::query_t constrained_rigid_body_query_;

         // Query for rigid bodies that are dynamic
         trecs::query_t dynamic_rigid_body_query_;

         // The raw constraint graph for this frame.
         DynamicArray<graph::types::edge_t<trecs::uid_t, edgeId_t> > constraint_graph_;

         // The connected component arrangement of the full constraint graph.
         // Individual connected components are denoted by 'bookmarks_'.
         DynamicArray<graph::types::labeled_edge_t<trecs::uid_t, edgeId_t> > decomposed_graph_;

         // Bookmarks for the locations and sizes of connected components in
         // the constraint graph.
         DynamicArray<graph::types::subgraph_bookmark_t> bookmarks_;

         // A mapping of one subgraph's rigid body entities (dynamic and non-
         // dynamic) to their indices in the subgraph's Jacobian.
         std::unordered_map<trecs::uid_t, int> body_entity_to_index_;

         void addEqualityConstraintsToGraph(
            const std::unordered_set<trecs::uid_t> & eq_constraint_edge_entities,
            const trecs::ComponentArrayWrapper<trecs::edge_t> & eq_constraint_edges,
            const trecs::ComponentArrayWrapper<oy::types::constraintOutput_t> & eq_constraint_outputs
         );

         void addInequalityConstraintsToGraph(
            const std::unordered_set<trecs::uid_t> & ineq_constraint_edge_entities,
            const trecs::ComponentArrayWrapper<std::vector<trecs::edge_t> > & ineq_constraint_edges,
            const trecs::ComponentArrayWrapper<ineqConstraintOutput_t> & ineq_constraint_outputs
         );

         void calculateBodyIndexMapping(
            const graph::Subgraph<trecs::uid_t, edgeId_t> & subgraph
         );

         void calculateBigConstraintMatrices(
            const graph::Subgraph<trecs::uid_t, edgeId_t> & subgraph,
            const trecs::ComponentArrayWrapper<oy::types::constraintOutput_t> & eq_constraint_outputs,
            const trecs::ComponentArrayWrapper<ineqConstraintOutput_t> & ineq_constraint_outputs,
            HeapMatrix & big_L,
            HeapMatrix & baumgartes,
            HeapMatrix & lambda_bounds
         ) const;

         void calculateBigBodyMatrices(
            const std::unordered_set<trecs::uid_t> & dynamic_body_entities,
            const trecs::ComponentArrayWrapper<oy::types::constrainedRigidBody_t> & constrained_rigid_bodies,
            HeapMatrix & big_M_inv,
            HeapMatrix & big_next_q_vel
         ) const;

         void applyImpulses(
            const HeapMatrix & big_L,
            const HeapMatrix & lambda_dt,
            const std::unordered_set<trecs::uid_t> & dynamic_rigid_body_entities,
            trecs::ComponentArrayWrapper<oy::types::rigidBody_t> & bodies
         ) const;

         void applyConstraints(
            const graph::Subgraph<trecs::uid_t, edgeId_t> & subgraph,
            const trecs::ComponentArrayWrapper<oy::types::constraintOutput_t> & eq_constraint_outputs,
            const trecs::ComponentArrayWrapper<ineqConstraintOutput_t> & ineq_constraint_outputs,
            const trecs::ComponentArrayWrapper<oy::types::constrainedRigidBody_t> & constrained_rigid_bodies,
            const std::unordered_set<trecs::uid_t> & dynamic_rigid_body_entities,
            trecs::ComponentArrayWrapper<oy::types::rigidBody_t> & bodies
         );

         // Helper function to calculate portions of the PGS A matrix.
         // Specifically calculates the inner product of:
         //    (j_vec_m(a) ^ T) * (K_a * j_vec_n(a))
         // where:
         //    'K_a' is the 6x6 inverse mass matrix associated with body A.
         //    'j_vec_m(a)' is the Jacobian of the Mth constraint with respect
         //       to the velocity components of body A.
         //    'j_vec_n(a)' is the Jacobian of the Nth constraint with respect
         //       to the velocity components of body A.
         // The ordering of constraints is the same as the ordering of the edge
         // list in the decomposed graph.
         float contractJmaKaJna(
            int constraint_m_index,
            int constraint_n_index,
            int body_index,
            const HeapMatrix & big_J,
            const HeapMatrix & big_M_inv
         ) const;

         // Builds the A matrix for projected Gauss Seidel faster than naive
         // matrix multiplications, but not as fast as possible. Builds the
         // upper triangle and copies off-diagonal values to the lower
         // triangle.
         // The naive matrix multiplication is:
         //    A = big_J * big_M_inv * (big_J.transpose())
         void calculatePgsA(
            const graph::Subgraph<trecs::uid_t, edgeId_t> & subgraph,
            const HeapMatrix & big_J,
            const HeapMatrix & big_M_inv,
            HeapMatrix & pgs_A
         ) const;
   };
}

#endif
