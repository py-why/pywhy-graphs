.. _classes:

****************************************
:mod:`pywhy_graphs.classes`: Graph types
****************************************

.. automodule:: pywhy_graphs.classes
    :no-members:
    :no-inherited-members:

Pywhy-graphs provides data structures and methods for storing causal graphs.

The classes heavily rely on NetworkX and follows a similar API. We are generally
"networkx-compliant" in the sense that we have similar outputs for a similar input
and similarly named functions in the graph API. However, we also extend the API
to account for various classes of graphs that are not covered in networkx.

The choice of graph class depends on the structure of the
graph you want to represent. For reference on concepts repeated
across the API, see :ref:`glossary`.

Which graph class should I use?
===============================

Note, that we do not implement a causal DAG without latent confounders, because
that can be represented with a `networkx.DiGraph` with acyclicity constraints.

+-------------------+----------------------------------+-----------------------+
| Pywhy_graph Class | Edge Types                       | Latent confounders    |
+===================+==================================+=======================+
| ADMG              | directed, bidirected, undirected | Yes                   |
+-------------------+----------------------------------+-----------------------+

We also represent common equivalence classes of causal graphs.

+-------------------+----------------------------------+-----------------------+
| Pywhy_graph Class | Edge Types                       | Latent confounders    |
+===================+==================================+=======================+
| CPDAG             | directed, undirected             | No                    |
+-------------------+----------------------------------+-----------------------+
| PAG               | directed, bidirected, undirected | Yes                   |
+-------------------+----------------------------------+-----------------------+

For representing interventions, we have an augmented graph, which stems from
the addition of "F-nodes", which represent interventions :footcite:`pearl1993aspects`.

+-------------------+------------------------------------------+--------------+
| Pywhy_graph Class | Edge Types                               | Known Target |
+===================+==========================================+==============+
| AugmentedGraph    | directed, undirected, bidirected         | Yes          |
+-------------------+------------------------------------------+--------------+
| IPAG              | directed, undirected, bidirected, circle | Yes          |
+-------------------+------------------------------------------+--------------+
| PsiPAG            | directed, undirected, bidirected, circle | No           |
+-------------------+------------------------------------------+--------------+

Finally, we also support time-series and create graphs that represent
stationary time-series causal processes.

+---------------------------------------+------------------------------------------+---------------------------------+
| Pywhy_graph Class                     | Edge Types                               | Analagous non time-series graph |
+=======================================+==========================================+=================================+
| StationaryTimeSeriesGraph             | undirected                               | nx.Graph                        |
+---------------------------------------+------------------------------------------+---------------------------------+
| StationaryTimeSeriesDiGraph           | directed                                 | nx.DiGraph                      |
+---------------------------------------+------------------------------------------+---------------------------------+
| StationaryTimeSeriesMixedEdgeGraph    | directed, undirected, bidirected         | MixedEdgeGraph                  |
+---------------------------------------+------------------------------------------+---------------------------------+
| StationaryTimeSeriesCPDAG             | directed, undirected                     | CPDAG                           |
+---------------------------------------+------------------------------------------+---------------------------------+
| StationaryTimeSeriesPAG               | directed, undirected, bidirected, circle | PAG                             |
+---------------------------------------+------------------------------------------+---------------------------------+


Causal graph types
==================

.. currentmodule:: pywhy_graphs.classes.timeseries
.. autoclass:: TimeSeriesGraph
    :noindex:
    :inherited-members:
.. autoclass:: TimeSeriesDiGraph
    :noindex:
    :inherited-members:
.. autoclass:: TimeSeriesMixedEdgeGraph
    :noindex:
    :inherited-members:
.. autoclass:: StationaryTimeSeriesCPDAG
    :noindex:
    :inherited-members:
.. autoclass:: StationaryTimeSeriesDiGraph
    :noindex:
    :inherited-members:
.. autoclass:: StationaryTimeSeriesGraph
    :noindex:
    :inherited-members:
.. autoclass:: StationaryTimeSeriesMixedEdgeGraph
    :noindex:
    :inherited-members:
.. autoclass:: StationaryTimeSeriesPAG
    :noindex:
    :inherited-members:

.. note:: NetworkX uses `dicts` to store the nodes and neighbors in a graph.
   So the reporting of nodes and edges for the base graph classes may not
   necessarily be consistent across versions and platforms; however, the reporting
   for CPython is consistent across platforms and versions after 3.6.
