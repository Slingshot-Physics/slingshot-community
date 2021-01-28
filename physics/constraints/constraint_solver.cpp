#include "constraint_solver.hpp"

#include "graph.hpp"
#include "rigidbody.hpp"
#include "subgraph.hpp"

#include <cassert>

namespace oy
{
   ConstraintSolver::ConstraintSolver(void)
      : stove_(10)
      , constraint_graph_(64)
      , decomposed_graph_(64)
      , bookmarks_(64)
   { }

   void ConstraintSolver::registerComponents(
      trecs::Allocator & allocator
   ) const
   {
      allocator.registerComponent<oy::types::constraintOutput_t>();
      allocator.registerComponent<oy::types::constrainedRigidBody_t>();
      allocator.registerComponent<std::vector<trecs::edge_t> >();
      allocator.registerComponent<oy::types::DynamicBody>();
      allocator.registerComponent<oy::types::rigidBody_t>();
   }

   void ConstraintSolver::registerQueries(trecs::Allocator & allocator)
   {
      eq_constraint_edge_query_= \
         allocator.addArchetypeQuery<
            oy::types::constraintOutput_t, trecs::edge_t
         >();

      eq_constraint_query_ = \
         allocator.addArchetypeQuery<oy::types::constraintOutput_t>();

      ineq_constraint_edge_query_ = \
         allocator.addArchetypeQuery<
            ineqConstraintOutput_t, std::vector<trecs::edge_t>
         >();

      ineq_constraint_query_ = \
         allocator.addArchetypeQuery<ineqConstraintOutput_t>();

      constrained_rigid_body_query_ = \
         allocator.addArchetypeQuery<oy::types::constrainedRigidBody_t>();

      dynamic_rigid_body_query_ = \
         allocator.addArchetypeQuery<
            oy::types::rigidBody_t, oy::types::DynamicBody
         >();
   }

   void ConstraintSolver::update(trecs::Allocator & allocator)
   {
      const auto eq_constraint_edges = allocator.getComponents<trecs::edge_t>();
      const auto eq_constraint_outputs = allocator.getComponents<oy::types::constraintOutput_t>();

      const auto ineq_constraint_edges = allocator.getComponents<std::vector<trecs::edge_t> >();
      const auto ineq_constraint_outputs = allocator.getComponents<ineqConstraintOutput_t>();

      auto bodies = allocator.getComponents<oy::types::rigidBody_t>();
      const auto constrained_bodies = allocator.getComponents<oy::types::constrainedRigidBody_t>();

      const auto & eq_constraint_edge_entities = \
         allocator.getQueryEntities(eq_constraint_edge_query_);
      const auto & ineq_constraint_edge_entities = \
         allocator.getQueryEntities(ineq_constraint_edge_query_);
      const auto & dynamic_body_entities = \
         allocator.getQueryEntities(dynamic_rigid_body_query_);

      constraint_graph_.clear();
      bookmarks_.clear();
      decomposed_graph_.clear();

      addEqualityConstraintsToGraph(
         eq_constraint_edge_entities,
         eq_constraint_edges,
         eq_constraint_outputs
      );

      addInequalityConstraintsToGraph(
         ineq_constraint_edge_entities,
         ineq_constraint_edges,
         ineq_constraint_outputs
      );

      graph::connectedComponentsDecomposition(
         constraint_graph_, decomposed_graph_, bookmarks_
      );

      for (const auto & bookmark : bookmarks_)
      {
         graph::Subgraph<trecs::uid_t, edgeId_t> subgraph(
            decomposed_graph_, bookmark
         );

         applyConstraints(
            subgraph,
            eq_constraint_outputs,
            ineq_constraint_outputs,
            constrained_bodies,
            dynamic_body_entities,
            bodies
         );
      }
   }

   void ConstraintSolver::addEqualityConstraintsToGraph(
      const std::unordered_set<trecs::uid_t> & eq_constraint_edge_entities,
      const trecs::ComponentArrayWrapper<trecs::edge_t> & eq_constraint_edges,
      const trecs::ComponentArrayWrapper<oy::types::constraintOutput_t> & eq_constraint_outputs
   )
   {
      for (const auto & entity : eq_constraint_edge_entities)
      {
         const trecs::edge_t & eq_constraint_edge = *eq_constraint_edges[entity];
         graph::types::edge_t<trecs::uid_t, edgeId_t> eq_edge;
         eq_edge.nodeIdA = eq_constraint_edge.nodeIdA;
         eq_edge.nodeIdB = eq_constraint_edge.nodeIdB;
         eq_edge.edgeId.entity = entity;
         // An offset of -1 indicates that this edge comes from an equality
         // constraint.
         eq_edge.edgeId.offset = -1;
         eq_edge.edgeId.constraintType = eq_constraint_outputs[entity]->constraintType;

         switch(eq_constraint_edge.flag)
         {
            case trecs::edge_flag::TRANSITIVE:
               eq_edge.edgeType = graph::types::enumTransitivity_t::TRANSITIVE;
               constraint_graph_.push_back(eq_edge);
               break;
            case trecs::edge_flag::NODE_A_TERMINAL:
               eq_edge.edgeType = graph::types::enumTransitivity_t::NODE_A_TERMINAL;
               constraint_graph_.push_back(eq_edge);
               break;
            case trecs::edge_flag::NODE_B_TERMINAL:
               eq_edge.edgeType = graph::types::enumTransitivity_t::NODE_B_TERMINAL;
               constraint_graph_.push_back(eq_edge);
               break;
            case trecs::edge_flag::NULL_EDGE:
               break;
         }
      }
   }

   void ConstraintSolver::addInequalityConstraintsToGraph(
      const std::unordered_set<trecs::uid_t> & ineq_constraint_edge_entities,
      const trecs::ComponentArrayWrapper<std::vector<trecs::edge_t> > & ineq_constraint_edges,
      const trecs::ComponentArrayWrapper<ineqConstraintOutput_t> & ineq_constraint_outputs
   )
   {
      // Each inequality constraint edge has a vector of constraint output
      // associated with it.
      for (const auto entity : ineq_constraint_edge_entities)
      {
         const std::vector<trecs::edge_t> & ineq_edges = *ineq_constraint_edges[entity];
         const ineqConstraintOutput_t & ineq_outputs = *ineq_constraint_outputs[entity];

         int i = 0;
         for (const auto & ineq_constraint_edge : ineq_edges)
         {
            graph::types::edge_t<trecs::uid_t, edgeId_t> ineq_edge;
            ineq_edge.nodeIdA = ineq_constraint_edge.nodeIdA;
            ineq_edge.nodeIdB = ineq_constraint_edge.nodeIdB;
            ineq_edge.edgeId.entity = entity;
            // Offset is used to mark the index of the vector for the
            // constraint output that produced this constraint edge.
            ineq_edge.edgeId.offset = i;
            ineq_edge.edgeId.constraintType = ineq_outputs[i].constraintType;

            switch(ineq_constraint_edge.flag)
            {
               case trecs::edge_flag::TRANSITIVE:
                  ineq_edge.edgeType = graph::types::enumTransitivity_t::TRANSITIVE;
                  constraint_graph_.push_back(ineq_edge);
                  break;
               case trecs::edge_flag::NODE_A_TERMINAL:
                  ineq_edge.edgeType = graph::types::enumTransitivity_t::NODE_A_TERMINAL;
                  constraint_graph_.push_back(ineq_edge);
                  break;
               case trecs::edge_flag::NODE_B_TERMINAL:
                  ineq_edge.edgeType = graph::types::enumTransitivity_t::NODE_B_TERMINAL;
                  constraint_graph_.push_back(ineq_edge);
                  break;
               case trecs::edge_flag::NULL_EDGE:
                  break;
            }

            ++i;
         }
      }
   }

   void ConstraintSolver::calculateBodyIndexMapping(
      const graph::Subgraph<trecs::uid_t, edgeId_t> & subgraph
   )
   {
      // Each rigid body entity in the constraint subgraph has an index in the
      // big Jacobian for that constraint subgraph. However, only "active"
      // bodies contribute to the big Jacobian (active bodies are bodies that
      // are Dynamic or Stationary and whose entity UIDs are >= 0).
      // An entity UID < 0 implies that a point in the "world" (or the "null
      // body") constrains a dynamic body.
      body_entity_to_index_.clear();
      body_entity_to_index_.reserve(2 * subgraph.size());

      unsigned int counter = 0;
      for (const auto & edge : subgraph)
      {
         trecs::uid_t node_id_a = edge.nodeIdA;
         if (
            (body_entity_to_index_.count(node_id_a) == 0) && (node_id_a >= 0)
         )
         {
            body_entity_to_index_[node_id_a] = counter;
            ++counter;
         }

         trecs::uid_t node_id_b = edge.nodeIdB;
         if (
            (body_entity_to_index_.count(node_id_b) == 0) && (node_id_b >= 0)
         )
         {
            body_entity_to_index_[node_id_b] = counter;
            ++counter;
         }
      }
   }

   void ConstraintSolver::calculateBigConstraintMatrices(
      const graph::Subgraph<trecs::uid_t, edgeId_t> & subgraph,
      const trecs::ComponentArrayWrapper<oy::types::constraintOutput_t> & eq_constraint_outputs,
      const trecs::ComponentArrayWrapper<ineqConstraintOutput_t> & ineq_constraint_outputs,
      HeapMatrix & big_L,
      HeapMatrix & baumgartes,
      HeapMatrix & lambda_bounds
   ) const
   {
      assert(big_L.num_rows == (6 * body_entity_to_index_.size()));
      assert(big_L.num_cols == subgraph.size());

      assert(baumgartes.num_rows == subgraph.size());
      assert(baumgartes.num_cols == 1);

      assert(lambda_bounds.num_rows == subgraph.size());
      assert(lambda_bounds.num_cols == 2);

      unsigned int i = 0;
      for (const auto & edge : subgraph)
      {
         const trecs::uid_t constraint_entity = edge.edgeId.entity;
         const int constraint_offset = edge.edgeId.offset;

         const oy::types::constraintOutput_t * temp_output = nullptr;

         // Constraint output can either come directly from components in the
         // ECS (equality constraint output), or it can come from a component
         // that has a *vector* of constraint output types (inequality
         // constraint output)
         if (constraint_offset < 0)
         {
            temp_output = eq_constraint_outputs[constraint_entity];
         }
         else
         {
            temp_output = \
               &ineq_constraint_outputs[constraint_entity]->at(constraint_offset);
         }

         const Matrix<12, 1> & jacobian = temp_output->jacobian;
         const float & lambda_min = temp_output->lambdaMin;
         const float & lambda_max = temp_output->lambdaMax;
         const float & baumgarte = temp_output->baumgarte;

         for (unsigned int j = 0; j < 12; ++j)
         {
            trecs::uid_t body_id = (j < 6) ? edge.nodeIdA : edge.nodeIdB;
            if (body_id != oy::types::null_body_entity)
            {
               unsigned int body_index = static_cast<unsigned int>(body_entity_to_index_.at(body_id));
               big_L(body_index * 6 + j - (j >= 6) * 6, i) = jacobian(j, 0);
            }
         }

         baumgartes(i, 0) = baumgarte;
         lambda_bounds(i, 0) = lambda_min;
         lambda_bounds(i, 1) = lambda_max;

         ++i;
      }
   }

   void ConstraintSolver::calculateBigBodyMatrices(
      const std::unordered_set<trecs::uid_t> & dynamic_body_entities,
      const trecs::ComponentArrayWrapper<oy::types::constrainedRigidBody_t> & constrained_rigid_bodies,
      HeapMatrix & big_M_inv,
      HeapMatrix & big_next_q_vel
   ) const
   {
      assert(big_M_inv.num_rows == 6);
      assert(big_M_inv.num_cols == (6 * body_entity_to_index_.size()));

      assert(big_next_q_vel.num_rows == (6 * body_entity_to_index_.size()));
      assert(big_next_q_vel.num_cols == 1);

      for (const auto entity : dynamic_body_entities)
      {
         if (body_entity_to_index_.count(entity) == 0)
         {
            continue;
         }

         int body_index = body_entity_to_index_.at(entity);

         const oy::types::constrainedRigidBody_t * constrained_rigid_body = \
            constrained_rigid_bodies[entity];

         for (unsigned int j = 0; j < 3; ++j)
         {
            big_M_inv(j + 0, j + body_index * 6) = \
               constrained_rigid_body->invMass;
         }

         const Matrix33 inertia_inv = constrained_rigid_body->invInertia;

         for (unsigned int j = 0; j < 3; ++j)
         {
            for (unsigned int k = 0; k < 3; ++k)
            {
               big_M_inv(j + 3, k + 3 + body_index * 6) = \
                  inertia_inv(j, k);
            }
         }

         for (unsigned int j = 0; j < 6; ++j)
         {
            big_next_q_vel(body_index * 6 + j + 0, 0) = \
               constrained_rigid_body->nextQVel(j, 0);
         }
      }
   }

   void ConstraintSolver::applyImpulses(
      const HeapMatrix & big_L,
      const HeapMatrix & lambda_dt,
      const std::unordered_set<trecs::uid_t> & dynamic_rigid_body_entities,
      trecs::ComponentArrayWrapper<oy::types::rigidBody_t> & bodies
   ) const
   {
      assert(big_L.num_rows == (6 * body_entity_to_index_.size()));
      assert(big_L.num_cols == lambda_dt.num_rows);
      assert(lambda_dt.num_cols == 1);

      HeapMatrix impulses(6 * body_entity_to_index_.size(), 1);
      for (unsigned int i = 0; i < big_L.num_rows; ++i)
      {
         impulses(i, 0) = big_L.innerProductRowVecSse(i, lambda_dt);
      }

      Vector3 temp_linear_impulse;
      Vector3 temp_angular_impulse;

      for (const auto entity : dynamic_rigid_body_entities)
      {
         if (body_entity_to_index_.count(entity) == 0)
         {
            continue;
         }

         unsigned int i = body_entity_to_index_.at(entity);
         for (unsigned int j = 0; j < 3; ++j)
         {
            temp_linear_impulse[j] = impulses(6 * i + j, 0);
            temp_angular_impulse[j] = impulses(6 * i + j + 3, 0);
         }

         oy::types::rigidBody_t & body = *bodies[entity];

         if (temp_linear_impulse.hasNan())
         {
           std::cout << "lin impulse for body: " << entity << " has nan\n";
         }

         if (temp_angular_impulse.hasNan())
         {
           std::cout << "ang impulse for body: " << entity << " has nan\n";
         }

         oy::rb::applyLinearImpulse(
            body, temp_linear_impulse, oy::types::enumFrame_t::GLOBAL
         );

         oy::rb::applyAngularImpulse(
            body, temp_angular_impulse, oy::types::enumFrame_t::GLOBAL
         );
      }
   }

   void ConstraintSolver::applyConstraints(
      const graph::Subgraph<trecs::uid_t, edgeId_t> & subgraph,
      const trecs::ComponentArrayWrapper<oy::types::constraintOutput_t> & eq_constraint_outputs,
      const trecs::ComponentArrayWrapper<ineqConstraintOutput_t> & ineq_constraint_outputs,
      const trecs::ComponentArrayWrapper<oy::types::constrainedRigidBody_t> & constrained_rigid_bodies,
      const std::unordered_set<trecs::uid_t> & dynamic_rigid_body_entities,
      trecs::ComponentArrayWrapper<oy::types::rigidBody_t> & bodies
   )
   {
      calculateBodyIndexMapping(subgraph);
      unsigned int num_bodies = body_entity_to_index_.size();
      unsigned int num_constraints = subgraph.size();

      // Calculate the big constraint matrices out of constraint output:
      //    big_L, big_J, baumgartes, lambda_bounds
      HeapMatrix big_L(6 * num_bodies, num_constraints);
      HeapMatrix baumgartes(num_constraints, 1);
      HeapMatrix lambda_bounds(num_constraints, 2);
      calculateBigConstraintMatrices(
         subgraph,
         eq_constraint_outputs,
         ineq_constraint_outputs,
         big_L,
         baumgartes,
         lambda_bounds
      );
      HeapMatrix big_J(big_L.transpose());

      HeapMatrix big_M_inv(6, 6 * num_bodies);
      HeapMatrix big_next_q_vel(6 * num_bodies, 1);
      calculateBigBodyMatrices(
         dynamic_rigid_body_entities,
         constrained_rigid_bodies,
         big_M_inv,
         big_next_q_vel
      );

      HeapMatrix A(num_constraints, num_constraints);
      calculatePgsA(subgraph, big_J, big_M_inv, A);
      HeapMatrix b(-1.f * (big_J * big_next_q_vel) - baumgartes);

      // Guard against zeroes on the diagonal of the A matrix to avoid
      // singularities.
      for (unsigned int i = 0; i < A.num_rows; ++i)
      {
         A(i, i) += (A(i, i) == 0.0f) * 1.0f;
      }

      HeapMatrix lambda_dt(num_constraints, 1);

      stove_.pull(lambda_dt, subgraph, 0.01f);

      ProjectedGaussSeidel(A, b, lambda_bounds, lambda_dt);

      stove_.push(lambda_dt, subgraph);

      applyImpulses(big_L, lambda_dt, dynamic_rigid_body_entities, bodies);
   }

   float ConstraintSolver::contractJmaKaJna(
      int constraint_m_index,
      int constraint_n_index,
      int body_index,
      const HeapMatrix & big_J,
      const HeapMatrix & big_M_inv
   ) const
   {
      float val = 0.f;
      for (unsigned int m = 0; m < 3; ++m)
      {
         float lin_vel_terms = big_M_inv.innerProductSubRowMatRowSse(
            m + 0, body_index * 6 + 0, constraint_n_index, body_index * 6 + 0, big_J, 3
         );
         float ang_vel_terms = big_M_inv.innerProductSubRowMatRowSse(
            m + 3, body_index * 6 + 3, constraint_n_index, body_index * 6 + 3, big_J, 3
         );

         val += big_J(constraint_m_index, body_index * 6 + m + 0) * lin_vel_terms;
         val += big_J(constraint_m_index, body_index * 6 + m + 3) * ang_vel_terms;
      }

      return val;
   }

   void ConstraintSolver::calculatePgsA(
      const graph::Subgraph<trecs::uid_t, edgeId_t> & subgraph,
      const HeapMatrix & big_J,
      const HeapMatrix & big_M_inv,
      HeapMatrix & pgs_A
   ) const
   {
      assert(big_J.num_rows == subgraph.size());
      assert(big_J.num_cols == (6 * body_entity_to_index_.size()));

      assert(big_M_inv.num_rows == 6);
      assert(big_M_inv.num_cols == (6 * body_entity_to_index_.size()));

      assert(pgs_A.num_rows == subgraph.size());
      assert(pgs_A.num_cols == subgraph.size());

      for (unsigned int i = 0; i < subgraph.size(); ++i)
      {
         const graph::types::labeled_edge_t<trecs::uid_t, edgeId_t> & edge = subgraph[i];

         trecs::uid_t constraint_body_entities[2] = {
            edge.nodeIdA, edge.nodeIdB
         };

         int body_indices[2] = {
            edge.nodeIdA >= 0 ? body_entity_to_index_.at(edge.nodeIdA) : oy::types::null_body_id,
            edge.nodeIdB >= 0 ? body_entity_to_index_.at(edge.nodeIdB) : oy::types::null_body_id
         };

         for (unsigned int b = 0; b < 2; ++b)
         {
            int body_index = body_indices[b];
            if (body_index == oy::types::null_body_id)
            {
               continue;
            }

            // Calculate on-diagonal elements
            pgs_A(i, i) += contractJmaKaJna(
               i, i, body_index, big_J, big_M_inv
            );

            // Calculate off-diagonal elements
            for (unsigned int j = 0; j < i; ++j)
            {
               // Look for another constraint edge in this connected component
               // that's connected to the current body ID.
               if (
                  (constraint_body_entities[b] == subgraph[j].nodeIdA) ||
                  (constraint_body_entities[b] == subgraph[j].nodeIdB)
               )
               {
                  pgs_A(i, j) += contractJmaKaJna(
                     i, j, body_index, big_J, big_M_inv
                  );
               }

               pgs_A(j, i) = pgs_A(i, j);
            }
         }
      }
   }
}
