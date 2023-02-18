.. _api_ref:

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

:mod:`pywhy_graphs.classes`: Causal graph classes
=================================================
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
   is_definite_noncollider

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

Timeseries
==========
The following are useful functions that operate specifically on time-series graphs.

.. currentmodule:: pywhy_graphs.classes.timeseries
.. autosummary::
   :toctree: generated/

   complete_ts_graph
   empty_ts_graph
   get_summary_graph
   has_homologous_edges
   nodes_in_time_order

We also have classes for representing causal time-series graphs.

Pywhy-graphs implements a networkx-like graph class for representing time-series.
Stationary causal timeseries graphs may be useful in various applications.

.. currentmodule:: pywhy_graphs.classes.timeseries
.. autosummary::
   :toctree: generated/
   
   TimeSeriesGraph
   TimeSeriesDiGraph
   TimeSeriesMixedEdgeGraph

For stationary time-series, we explicitly represent them with different classes.

.. autosummary::
   :toctree: generated/

   StationaryTimeSeriesCPDAG
   StationaryTimeSeriesDiGraph
   StationaryTimeSeriesGraph
   StationaryTimeSeriesMixedEdgeGraph
   StationaryTimeSeriesPAG

:mod:`pywhy_graphs.simulate`: Causal graphical model simulations
================================================================
Pywhy-graphs implements a various functions for assisting in simulating
a SCM and their data starting from the causal graph.

.. currentmodule:: pywhy_graphs

.. autosummary::
   :toctree: generated/

   simulate.simulate_linear_var_process
   simulate.simulate_data_from_var
   simulate.simulate_var_process_from_summary_graph


Visualization of causal graphs
==============================
Visualization of causal graphs is different compared to networkx because causal graphs
can consist of mixed-edges. We implement an API that wraps ``graphviz`` and ``pygraphviz``
to perform modular visualization of nodes and edges.

.. currentmodule:: pywhy_graphs.viz

.. autosummary::
   :toctree: generated/

   draw
   timeseries_layout

Utilities for debugging
=======================
.. currentmodule:: pywhy_graphs

.. autosummary::
   :toctree: generated/

   sys_info

Simulation
==========
.. toctree::
   :maxdepth: 1

   reference/simulation/index
