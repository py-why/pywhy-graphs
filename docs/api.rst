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

   StructuralCausalModel
   CPDAG
   ADMG
   PAG

To see a breakdown of different inner graph functionalities,
see the :ref:`Graph API <graph_api>` page. See 

.. toctree::
   :maxdepth: 0

   graph_api


IO for reading/writing causal graphs
====================================
We advocate for using our implemented causal graph classes whenever
utilizing various packages. However, we also support transformations
to and from popular storage classes, such as ``numpy arrays``,
``pandas dataframes``, ``pgmpy``, ``DOT`` and ``dagitty``. Note that
not all these are supported for all types of graphs because of
inherent limitations in supporting mixed-edge graphs in other formats.

.. currentmodule:: pywhy_graphs.io

.. autosummary::
   :toctree: generated/

   load_from_networkx
   load_from_numpy
   load_from_pgmpy
   to_networkx
   to_numpy

Converting Graphs
=================
.. currentmodule:: pywhy_graphs.algorithms

.. autosummary::
   :toctree: generated/

   dag2cpdag
   admg2pag

Utility Algorithms for Causal Graphs
====================================
.. currentmodule:: pywhy_graphs.algorithms

.. autosummary::
   :toctree: generated/

   discriminating_path
   possibly_d_sep_sets
   uncovered_pd_path
   is_markov_equivalent
   compute_v_structures