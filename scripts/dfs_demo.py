from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

@dataclass
class edge_t:
   node_id_a: int
   node_id_b: int
   component_uid: int

@dataclass
class node_t:
   id: int
   component_uid: int

def dfs(
   edges: List[edge_t],
   node_markers: List[int],
   component_uid: int,
   start_node: node_t
) -> List[edge_t]:
   frontier: List[int] = []
   frontier.append(start_node.id)

   # This should have all of the edges from a single connected component
   marked_edges: List[edge_t] = []

   while len(frontier) > 0:
      print("frontier:", frontier)
      node_id = frontier.pop(0)
      while node_markers[node_id] != 0:
         print("\tnode id marker: ", node_markers[node_id])
         if len(frontier) > 0:
            node_id = frontier.pop(0)
         else:
            node_id = -1
            break

      if node_id == -1:
         break

      print("node id: ", node_id)
      node_markers[node_id] = component_uid

      remove_edge_indices = []
      for i, edge in enumerate(edges):
         if edge.node_id_a == node_id:
            print("linking edge found:", edge.node_id_a, edge.node_id_b)
            remove_edge_indices.append(i)
            frontier.append(edge.node_id_b)
         elif edge.node_id_b == node_id:
            print("linking edge found:", edge.node_id_a, edge.node_id_b)
            remove_edge_indices.append(i)
            frontier.append(edge.node_id_a)

      # Reversing the index order from highest to lowest gets rid of index
      # invalidation during deletion.
      for removed_edge_index in sorted(remove_edge_indices, reverse=True):
         marked_edges.append(edges.pop(removed_edge_index))

   return marked_edges

def main():
   temp_edge_list = [
      edge_t(2, 3, 0),
      edge_t(4, 3, 0),
      edge_t(5, 4, 0),
      edge_t(10, 8, 0),
      edge_t(11, 7, 0),
      edge_t(5, 7, 0),
      edge_t(8, 2, 0),
      edge_t(10, 11, 0),
      edge_t(0, 1, 0),
      edge_t(6, 1, 0),
      edge_t(6, 9, 0),
      edge_t(5, 8, 0),
      edge_t(2, 4, 0),
      edge_t(5, 3, 0),
      edge_t(7, 3, 0),
   ]

   G = nx.Graph()
   G.add_edges_from([(edge.node_id_a, edge.node_id_b) for edge in temp_edge_list])
   plt.figure()
   nx.draw_networkx(G)
   plt.show()

   print("original edge list")
   for edge in temp_edge_list:
      print(edge.node_id_a, edge.node_id_b)

   node_markers = [0 for _ in range(12)]

   connected_component = dfs(temp_edge_list, node_markers, 1, node_t(3, 0))

   for edge in connected_component:
      print(edge.node_id_a, edge.node_id_b)

   connected_component = dfs(temp_edge_list, node_markers, 1, node_t(0, 0))

   for edge in connected_component:
      print(edge.node_id_a, edge.node_id_b)

if __name__ == "__main__":
   main()
