.. _export:

**************************************************************************************
Importing causal graphs to PyWhy-Graphs, or exporting PyWhy-Graphs to another package
**************************************************************************************

.. automodule:: pywhy_graphs.export
    :no-members:
    :no-inherited-members:

Pywhy-graphs provides light-weight data structures and networkx-like methods for
storing causal graphs.

The causality community is quite large and currently there is no de-facto standard
for representing causal graphs with mixed edges. Therefore, we provide an interface
for importing/exporting graphs from other packages.

We currently only offer support for a package if they have a way of representing
all types of causal graphs (not just DAGs). We welcome contributions here!

Causal-Learn
============
Causal-learn maintains its own graph interpretation in the form of a structured
upper-triangular numpy array.

.. currentmodule:: pywhy_graphs.export
   
.. autosummary::
   :toctree: ../../generated/

   export.graph_to_clearn_arr
   export.clearn_arr_to_graph


Numpy (graphviz and dagitty)
============================
GraphViz stores a graph that allows mixed-edges using a numpy array with values filled
in following a specific enumeration, so all values can be mapped to an edge, or multiple
edges if more than one edge is allowed between nodes.

.. autosummary::
   :toctree: ../../generated/

   export.graph_to_numpy
   export.numpy_to_graph


PCAlg from R (Experimental)
===========================
``pcalg`` from R uses the following mapping for their array:

- 0: no edge
- 1: -o
- 2: -> (arrowhead)
- 3: - (tail)

and stores a structured numpy array following this format. This functionality
is experimental because this is not tested against the actual implementation in R.
Please raise an issue if you encounter errors, or issues.

.. autosummary::
   :toctree: ../../generated/

   export.graph_to_pcalg
   export.pcalg_to_graph
