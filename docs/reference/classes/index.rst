.. _classes:

**********************
Causal Graphs in PyWhy
**********************

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

+------------------------------------+----------------------------------+---------------------------------+
| Pywhy_graph Class                  | Edge Types                       | Analagous non time-series graph |
+====================================+==================================+=================================+
| StationaryTimeSeriesGraph          | undirected                       | nx.Graph                        |
+------------------------------------+----------------------------------+---------------------------------+
| StationaryTimeSeriesDiGraph        | directed                         | nx.DiGraph                      |
+------------------------------------+----------------------------------+---------------------------------+
| StationaryTimeSeriesMixedEdgeGraph | directed, undirected, bidirected | MixedEdgeGraph                  |
+------------------------------------+----------------------------------+---------------------------------+
| StationaryTimeSeriesCPDAG          | directed, undirected             | CPDAG                           |
+------------------------------------+----------------------------------+---------------------------------+
| StationaryTimeSeriesPAG            | directed, bidirected, circle     | PAG                             |
+------------------------------------+----------------------------------+---------------------------------+

:mod:`pywhy_graphs.classes`: Causal graph types
===============================================
.. currentmodule:: pywhy_graphs.classes
    
.. autoclass:: ADMG
    :noindex:
    :inherited-members:
.. autoclass:: CPDAG
    :noindex:
    :inherited-members:
.. autoclass:: PAG
    :noindex:
    :inherited-members:

:mod:`pywhy_graphs.classes.timeseries`: Causal graph types for time-series (alpha)
==================================================================================
.. automodule:: pywhy_graphs.classes.timeseries
   :no-members:
   :no-inherited-members:

Currently, we have an alpha support for time-series causal graphs. This means that their internals
and API will most surely change over the next few versions.

Support of time-series is implemented in the form of more structured graph classes, where
every graph has two major differences:

- max lag: Every graph has a keyword argument input ``max_lag``, specifying the maximum lag
  that the time-series graph can represent.
- time-series node (tsnode): Every graph's nodes are required to be a 2-tuple, with the variable
  name as the first element and the lag as the second element.
- time-ordered: All edges are time-ordered, unless the underlying graph is an undirected
  :class:`networkx.Graph`. Time-ordered edges means that there are no directed edges pointing from
  the present to the past, so there are no edges of the form ``(('x', -t), ('y', -t'))``, where
  ``t < t'``. For example, a directed edge of the form ``(('x', -3), ('y', -4))`` is not allowed.
- selection bias (undirected edges): There is no support for undirected edges, or selection bias
  in time-series causal graphs at this moment.

Some graphs also embody the implicit assumption of "stationarity", which means all edges are
repeated over time. For example: if we assume stationarity, and know the edge
``(('x', -3), ('y', -2))`` exists in the graph and the maximum lag is 4, then the following edges
also exist in the graph:

- ``(('x', -4), ('y', -3))``
- ``(('x', -2), ('y', -1))``
- ``(('x', -1), ('y', 0))``

Stationarity implies that all edge additions/removals propagate to other homologous edges :footcite:`entner2010tsfci`.
This property can be turned off in :class:`~pywhy_graphs.classes.timeseries.StationaryTimeSeriesCPDAG` and
:class:`~pywhy_graphs.classes.timeseries.StationaryTimeSeriesPAG` by calling the `set_stationarity` function.
This may be useful for example in causal discovery, where we are modifying edges, but do not want
the modifications to propagate to homologous edges.

Note that stationarity in the Markov equivalence class of the causal graphs has some subtle differences
that impact the causal assumptions encoded in the MEC. All other functionalities are similar. See
:footcite:`gerhardus2021characterization` for a characterization of assumptions within a time-series
causal graph.

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

