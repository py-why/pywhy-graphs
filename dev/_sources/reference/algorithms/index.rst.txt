.. _algorithms:

********************************
Causal Graph Algorithms in PyWhy
********************************

.. automodule:: pywhy_graphs.algorithms
    :no-members:
    :no-inherited-members:


Pywhy-graphs provides data structures and methods for storing causal graphs, which
are documented in :ref:`classes`. We also provide a submodule for common graph
algorithms in the form of functions that take a mixed-edge graph as input.

Core Algorithms
---------------
.. currentmodule:: pywhy_graphs.algorithms
    
.. autosummary::
   :toctree: ../../generated/

   is_valid_mec_graph
   possible_ancestors
   possible_descendants
   discriminating_path
   is_definite_noncollider

.. currentmodule:: pywhy_graphs.networkx

.. autosummary::
   :toctree: ../../generated/

   bidirected_to_unobserved_confounder
   m_separated
   is_minimal_m_separator
   minimal_m_separator
   

Algorithms for Markov Equivalence Classes
-----------------------------------------
.. currentmodule:: pywhy_graphs.algorithms
.. autosummary::
   :toctree: ../../generated/

   pds
   pds_path
   uncovered_pd_path

Algorithms for Time-Series Graphs
---------------------------------

.. autosummary::
   :toctree: ../../generated/

   pds_t
   pds_t_path

Algorithms for handling acyclicity
----------------------------------

.. autosummary::
   :toctree: ../../generated/

   acyclification


***************************************
Semi-directed (possibly-directed) Paths
***************************************

.. automodule:: pywhy_graphs.algorithms.semi_directed_paths
.. autosummary::
   :toctree: ../../generated/

   all_semi_directed_paths
   is_semi_directed_path
