"""
======================================================
An introduction to Inducing Paths and how to find them
======================================================

A path p is called an ``inducing path`` relateve to <L,S>
on an ancestral graph, where every non-endpoint vertex on p
is either in L or a collider, and every collider on p is an ancestor
of either X , Y or a member of S.


In other words, it is a path between two nodes that cannot be
d-seperated, making it active regardless of what variables
we condition on.
"""

import pywhy_graphs
from pywhy_graphs import PAG
from pywhy_graphs.viz import draw

# construct a causal graph that will result in
# X <- Y <-> Z <-> H; Z -> X
G = PAG()
G.add_edge("X4", "X1", G.directed_edge_name)
G.add_edge("X2", "X5", G.directed_edge_name)
G.add_edge("X2", "X6", G.directed_edge_name)
G.add_edge("X4", "X6", G.directed_edge_name)
G.add_edge("X3", "X6", G.directed_edge_name)
G.add_edge("X3", "X4", G.directed_edge_name)
G.add_edge("X3", "X2", G.directed_edge_name)
G.add_edge("X5", "X6", G.directed_edge_name)
G.add_edge("X2", "X1", G.bidirected_edge_name)
G.add_edge("X4", "X5", G.bidirected_edge_name)
G.add_edge("X2", "X3", G.circle_edge_name)
G.add_edge("X4", "X3", G.circle_edge_name)
G.add_edge("X6", "X4", G.circle_edge_name)
G.add_edge("X6", "X5", G.circle_edge_name)


# X2 is the only collider on the path
L = {}
S = {"X2"}


# returns true
print(pywhy_graphs.inducing_path(G, "X1", "X3", L, S))
