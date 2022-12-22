###
API
###

:py:mod:`pywhy_graphs`:

.. automodule:: pywhy_graphs
   :no-members:
   :no-inherited-members:

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of pywhy-graphs, grouped thematically by analysis
stage.

Most-used classes
=================
These are the causal classes for Structural Causal Models (SCMs), or various causal
graphs encountered in the literature. 

.. currentmodule:: pywhy_graphs

.. autosummary::
   :toctree: generated/

   ADMG
   CPDAG
   PAG

Algorithms for Markov Equivalence Classes
=========================================
Traditional graph algorithms operate over graphs with only one type of edge.
Equivalence class graphs in causality generally consist of more than one type of
edge. These algorithms are common algorithms used in a variety of different
causal graph operations.

.. currentmodule:: pywhy_graphs.algorithms

.. autosummary::
   :toctree: generated/

   is_valid_mec_graph
   possible_ancestors
   possible_descendants
   discriminating_path
   pds
   pds_path
   uncovered_pd_path
   acyclification

Conversions between other package's causal graphs
=================================================
Other packages, such as `causal-learn <https://github.com/cmu-phil/causal-learn>`_,
implement various causal inference procedures, but encode a causal graph object
differently. This submodule converts between those causal graph data structures
and corresponding causal graphs in pywhy-graphs.

.. currentmodule:: pywhy_graphs.array

.. autosummary::
   :toctree: generated/

   graph_to_arr
   clearn_arr_to_graph

NetworkX Experimental Functionality
===================================
Currently, NetworkX does not support mixed-edge graphs, which are crucial
for representing causality with latent confounders and selection bias. The
following represent functionality that we intend to PR eventually into
networkx. They are included in pywhy-graphs as a temporary bridge. We 
welcome feedback.

.. currentmodule:: pywhy_graphs.networkx
.. autosummary::
   :toctree: generated/
   
   MixedEdgeGraph
   bidirected_to_unobserved_confounder
   m_separated
   
Visualization of causal graphs
==============================
Visualization of causal graphs is different compared to networkx because causal graphs
can consist of mixed-edges. We implement an API that wraps ``graphviz`` and ``pygraphviz``
to perform modular visualization of nodes and edges.

.. currentmodule:: pywhy_graphs.viz

.. autosummary::
   :toctree: generated/

   draw
