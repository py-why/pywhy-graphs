"""
.. _ex-selection-diagrams:

=========================================================
An introduction to selection diagrams and how to use them
=========================================================

Selection diagrams are causal graphical objects that allow the user and scientist
to represent causal models with multiple domains. This is useful for representing
domain-shifts, generalizability and invariances across different environments.
For a detailed theoretical introduction to selection diagrams, see
:footcite:`bareinboim_causal_2016,pearl2022external`.

This is a common problem in machine learning, where the goal is to learn a model
that generalizes to unseen data. In this case, the unseen data can be a different
domain, and the model needs to be invariant across domains.

This short example will introduce selection diagrams, and how they are constructed
and different from regular causal graphs.
"""

# %%
# Import the required libraries
# -----------------------------
from pprint import pprint

import pywhy_graphs as pg
from pywhy_graphs.algorithms import compute_invariant_domains_per_node, remove_snode_edge
from pywhy_graphs.viz import draw

# %%
# Build a selection diagram
# -------------------------
# Let us assume that there are only two domains in our causal model.
#
# A selection diagram fundamentally represents two different SCMs that represent
# the two different domains, but share some common variables and causal structure.
# Let M1 and M2 represent two different SCMs. Each SCM is a 4-tuple of the functionals,
# endogenous (observed) variables, exogenous (latent) variables and the probability
# distribution over the exogenous variables.
#
# :math:`M1 = \langle \mathcal{F}, V, U, P(u) \rangle`
# .. math::
#     V = \{W, X, Y, Z\}
#     P(U) = P(U_W, U_X, U_Y, U_Z)
#     \mathcal{F} = \begin{cases}
#           W = f_W(U_W) \\
#           X = f_X(U_X) \\
#           Y = f_Y(W, X, U_Y) \\
#           Z = f_Z(X, Y, U_Z)
#       \end{cases}
#
# :math:`M2 = \langle \mathcal{F'}, V, U', P'(u) \rangle`
# .. math::
#     P(U') = P(U_W', U_X', U_Y', U_Z')
#     \mathcal{F'} = \begin{cases}
#           W = f'_W(U_W) \\
#           X = f'_X(U_X) \\
#           Y = f'_Y(W, X, U_Y) \\
#           Z = f'_Z(X, Y, U_Z)
#       \end{cases}
#
# These two SCMs share the same causal structure, but the mechanisms for generating
# each variable may be different either due to different distributions over the
# exogenous variables, or different functional forms. The selection diagram encodes
# this information via an extra node, called the S-node, which represents the possibility
# of a difference in the data-generating mechanisms for the nodes it points to. The
# lack of an S-node pointing to a variable indicates that the data-generating mechanism
# for that variable is the same, or invariant across the two domains. This notion can
# be extended to N domains, where there are now :math:`\binom{N}{2}` S-nodes.
#
# The most general version of a selection diagram allows S-nodes to represent a
# change in graphical structure. We do not explore that generality in this example,
# or package :footcite:`pearl2022external`.
#
# We will now construct the selection diagram representing the two SCMs above.

# %%

G = pg.AugmentedGraph()
G.add_edges_from(
    [
        ("W", "Y"),
        ("X", "Y"),
        ("X", "Z"),
        ("Y", "Z"),
    ],
    edge_type=G.directed_edge_name,
)
G.add_s_node(domain_ids=(1, 2), node_changes=["W", "X", "Y", "Z"])
G.add_s_node(domain_ids=(2, 3), node_changes=["W", "X", "Y", "Z"])
G.add_s_node(domain_ids=(1, 3), node_changes=["W", "X", "Y", "Z"])

draw(G)

# %%
# Imposing cross-domain invariances
# ---------------------------------
# The selection diagram above allows for the possibility of different data-generating
# mechanisms for each variable. Currently, the S-nodes points to every single
# node in the graph. Therefore, there is no invariance across domains. Simply put,
# the data-generating mechanisms for each variable can be different across domains.
#
# However, we may want to impose invariances across domains 1 and 2 for the variables
# W and X. This can be done by removing the S-node pointing to W and X corresponding
# to domain 1 and 2.

# first, get the mapping from domain ids to s-nodes
domain_id_to_s_node = G.domain_ids_to_snodes

# remove the edge S^{1, 2} -> W
G = remove_snode_edge(G, domain_id_to_s_node[frozenset(1, 2)], "W")
G = remove_snode_edge(G, domain_id_to_s_node[frozenset(1, 2)], "X")

draw(G)

# let's explicitly compute the invariant domains per node
G = compute_invariant_domains_per_node(G, "W")
pprint(G.nodes(data=True))

# %%
# Consistency in cross-domain invariances
# ---------------------------------------
# In :footcite:`li2023discovery`, it is noted that there may be inconsistencies
# when removing S-node edges. For example, if we remove the edge S^{1, 2} -> W,
# and then remove the edge S^{2, 3} -> W, then we should have removed the
# edge S^{1, 3} -> W. This is because the invariances are transitive. In pywhy-graphs,
# we have a function that automatically checks for these inconsistencies and removes them.
# The :func:`pywhy_graphs.algorithms.remove_snode_edge` function automatically does this.

G = remove_snode_edge(G, domain_id_to_s_node[frozenset(2, 3)], "W")

# now the S-node edge corresponding to S^{1, 3} -> W should be removed as well
draw(G)

# %%
# Summary
# -------
# In this example, we have seen how to construct a selection diagram. We have also
# seen how to model invariances across domains using S-nodes and the lack of S-node edges
# to certain nodes in the graph.


# %%
# References
# ----------
# .. footbibliography::
