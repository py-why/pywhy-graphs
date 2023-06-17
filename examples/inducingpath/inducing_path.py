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
from pywhy_graphs import ADMG
from pywhy_graphs.viz import draw

# construct a causal graph that will result in
# X <- Y <-> Z <-> H; Z -> X
G = ADMG()
G.add_edge("Y", "X", G.directed_edge_name)
G.add_edge("Z", "X", G.directed_edge_name)
G.add_edge("Z", "Y", G.bidirected_edge_name)
G.add_edge("Z", "H", G.bidirected_edge_name)


dot_graph = draw(G)
dot_graph.render(outfile="admg.png", view=True)


# L contains the list of non-colliders in the path
L = {"Y"}

# Since the graph doesn't have a collider which is not
# an ancestor of any of the end-points, S is empty.
S = {}


print(pywhy_graphs.inducing_path(G, "X", "H", L, S))


# Construct a causal graph that will result in:
# X <-> Y <-> Z <-> H; Z -> X
G = ADMG()
G.add_edge("Y", "X", G.bidirected_edge_name)
G.add_edge("Z", "X", G.directed_edge_name)
G.add_edge("Z", "Y", G.bidirected_edge_name)
G.add_edge("Z", "H", G.bidirected_edge_name)

# There are no non-colliders in the path.
L = {}

# Y is a collider that not an ancestor of X or Y.
S = {"Y"}


print(pywhy_graphs.inducing_path(G, "X", "H", L, S))
