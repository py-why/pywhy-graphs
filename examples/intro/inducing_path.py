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

# %%
# Construct an example graph
# ----------------------------------------------
# To illustrate the workings of the inducing path algorithm, we will
# construct the causal graph from figure 2 of :footcite:`Colombo2012`.


G = PAG()
G.add_edge("X4", "X1", G.directed_edge_name)
G.add_edge("X2", "X5", G.directed_edge_name)
G.add_edge("X2", "X6", G.directed_edge_name)
G.add_edge("X4", "X6", G.directed_edge_name)
G.add_edge("X3", "X6", G.directed_edge_name)
G.add_edge("X3", "X4", G.directed_edge_name)
G.add_edge("X3", "X2", G.directed_edge_name)
G.add_edge("X5", "X6", G.directed_edge_name)
G.add_edge("L2", "X4", G.directed_edge_name)
G.add_edge("L2", "X5", G.directed_edge_name)
G.add_edge("L1", "X1", G.directed_edge_name)
G.add_edge("L1", "X2", G.directed_edge_name)
G.add_edge("X2", "X3", G.circle_edge_name)
G.add_edge("X4", "X3", G.circle_edge_name)
G.add_edge("X6", "X4", G.circle_edge_name)
G.add_edge("X6", "X5", G.circle_edge_name)


# this is the Figure 2(a) in the paper as we see.
dot_graph = draw(G)
dot_graph.render(outfile="pag.png", view=True)


# %%
# Adjacent nodes trivially have an inducing path
# ----------------------------------------------
# By definition, all adjacent nodes have a trivial inducing path between them,
# that path only consists of one edge, which is the edge between those two nodes.

L = {}
S = {}

# All such tests will return True.
print(pywhy_graphs.inducing_path(G, "X1", "X4", L, S))
print(pywhy_graphs.inducing_path(G, "X3", "X2", L, S))

# %%
# Inducing paths between non-adjacent nodes
# ---------------------------------------------
# Given the definition of an inducing path, we need to satisfy all
# requirements for the function to return True. Adding the latent
# variables to L is not enough for the pair [X1,X5]

L = {"L1", "L2"}
S = {}


# returns False
print(pywhy_graphs.inducing_path(G, "X1", "X5", L, S))


# However, if we add X3, a non-collider on the path
# from X1 to X5 to L, we open up an inducing path.


L = {"L1", "L2", "X3"}
S = {}

# now it returns True
print(pywhy_graphs.inducing_path(G, "X1", "X5", L, S))


# %%
# The Role of Colliders
# ----------------------------------------------
# Adding colliders to the set S has a downstream effect.
# Conditioning on a collider, or descendant of a collider opens up that collider path.
# For example, we will add the node 'X6' to the set ``S``. This will open up the collider
# path ``(X1, X2, X3)``, since 'X2' is an ancestor of 'X6'.

# Even now, some inducing paths are not opened.
# For example, the path between X1 and X3 is not available

# this returns False
print(pywhy_graphs.inducing_path(G, "X1", "X3", L, S))

# We need to add X6, which will open up paths
# including all the collider ancestors of X6
# in this case that node is X2.

L = {"L1", "L2", "X3"}
S = {"X6"}

# now it returns True
print(pywhy_graphs.inducing_path(G, "X1", "X3", L, S))
