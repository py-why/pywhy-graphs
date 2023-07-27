"""
.. _ex-selection-diagrams:

=========================================================
An introduction to selection diagrams and how to use them
=========================================================

Selection diagrams are causal graphical objects that allow the user and scientist
to represent causal models with multiple domains. This is useful for representing
domain-shifts, generalizability and invariances across different environments.

This is a common problem in machine learning, where the goal is to learn a model
that generalizes to unseen data. In this case, the unseen data can be a different
domain, and the model needs to be invariant across domains.

This short example will introduce selection diagrams, and how they are constructed
and different from regular causal graphs.
"""

import matplotlib.pyplot as plt
import networkx as nx

# %%
# Import the required libraries
# -----------------------------
import numpy as np

import pywhy_graphs as pg

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
#
#
# The most general version of a selection diagram allows S-nodes to represent a
# change in graphical structure. We do not explore that generality in this example,
# or package :footcite:`pearl2022external`.
