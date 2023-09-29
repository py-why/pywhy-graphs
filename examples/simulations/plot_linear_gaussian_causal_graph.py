"""
.. _ex-linear-gaussian-graph:

=====================================================
Linear Gaussian Graphs and Generating Continuous Data
=====================================================

Linear gaussian graphs are an important model. These are joint distributions
that follow the structure of a causal graph, where exogenous noise distributions are Gaussian
and nodes are linear combinations of their parents perturbed by the exogenous variable.

Thus, each edge is associated with a weight of how the parent node is added to the current
node.

In this example, we illustrate how to generate continuous data from a linear gaussian
causal graph.

For information on generating discrete data from a causal graph, one can see
:ref:`ex-discrete-cbn`. Consider reading the user-guide, :ref:`functional-causal-graphical-models`
to understand how an arbitrary functional relationships are encoded in a causal graph.
"""
