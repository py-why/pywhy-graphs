**pywhy-graphs**
===================

pywhy-graphs is a Python package for representing causal graphs. For example, Acyclic
Directed Mixed Graphs (ADMG), also known as causal DAGs and Partial Ancestral Graphs (PAGs).
We build on top of ``networkx's`` ``MixedEdgeGraph`` such that we maintain all the well-tested and efficient
algorithms and data structures of ``networkx``. 

We encourage you to use the package for your causal inference research and also build on top
with relevant Pull Requests. Also, see our `contributing guide <https://github.com/mne-tools/mne-icalabel/blob/main/CONTRIBUTING.md>`_.

Please refer to our :ref:`user_guide` for details on all the tools that we
provide. You can also find an exhaustive list of the public API in the
:ref:`api_ref`. You can also look at our numerous :ref:`examples <general_examples>`
that illustrate the use of ``pywhy_graphs`` in many different contexts.

API Stability
-------------
Currently, we are in very early stages of development. Most likely certain aspects of the causal
graphs API will change, but we will do our best to maintain some consistency. Our goal is to
eventually converge to a stable API that maintains the common software engineering release cycle traits
(e.g. deprecation cycles and API stability within major versions). Certain functionality
will be marked as "alpha" indicating that their might be drastic changes over different releases.
One should use alpha functionality with caution.

How do we compare with NetworkX?
--------------------------------
We fashioned pywhy-graphs API based on NetworkX because NetworkX's API is stable, robust and
has an existing community around it. Therefore, we expect all NetworkX users to have a relatively
low learning curve when transitioning to pywhy-graphs. However, NetworkX does not currently support
graphs with mixed-edges, so that is where the fundamental difference between the two APIs lie.

In all the "NetworkX-like" functions related to edges, such as ``add_edge``, ``has_edge``,
``number_of_edges``, etc. all have an additional keyword parameter, ``edge_type``. The edge
type is specified by this parameter and internally, all pywhy-graphs graph classes are a
composition of different networkx base graphs, :class:`networkx.DiGraph` and :class:`networkx.Graph`
that map to a user-specified edge type. For example,

.. code-block:: Python

   import pywhy_graphs.networkx as pywhy_nx
   import networkx as nx

   # a mixed-edge graph is a composition of networkx graphs
   # so we can create a representation of a graph with directed and
   # bidirected edges using the MixedEdgeGraph class
   G = pywhy_nx.MixedEdgeGraph()
   G.add_edge_type(nx.DiGraph(), edge_type='directed')
   G.add_edge_type(nx.Graph(), edge_type='bidirected')

   # when we use networkx-like API, we usually will have to specify the edge type
   G.add_edge(0, 1, edge_type='directed')

Because of this feature, not all NetworkX algorithms will work with pywhy-graphs because
they implicitly assume a single edge type. We implement common graph algorithms for
mixed-edge graphs that have utility in causal inference in the :mod:`pywhy_graphs.algorithms`
submodule. This is a similar design to NetworkX, where all graph classes have a relatively
lightweight API designed solely for interfacing with nodes and edges, while complex traversal
algorithms live in separate functions.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation
   Reference API<api>
   Usage (Simple Examples)<use>
   User Guide<user_guide>
   whats_new

Team
----

**pywhy-graphs** is developed and maintained by open-source contributors like yourself, and is always interested
in getting contributions from YOU! Our Github Issues page contains many issues that need to be solved to
improve the overall package. If you are interested in contributing, please do not hesitate to reach out to us on Github!

See our `contributing document <https://github.com/py-why/pywhy-graphs/CONTRIBUTING.md>`_ for details on our approach to Issues and Pull Requests.

To learn more about who specifically contributed to this codebase, see
`our contributors <https://github.com/py-why/pywhy-graphs/graphs/contributors>`_ page.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
