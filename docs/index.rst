**pywhy-graphs**
===================

pywhy-graphs is a Python package for representing causal graphs. For example, Acyclic
Directed Mixed Graphs (ADMG), also known as causal DAGs and Partial Ancestral Graphs (PAGs).
We build on top of ``networkx's`` ``MixedEdgeGraph`` such that we maintain all the well-tested and efficient
algorithms and data structures of ``networkx``. 

We encourage you to use the package for your causal inference research and also build on top
with relevant Pull Requests. Also, see our `contributing guide <https://github.com/mne-tools/mne-icalabel/blob/main/CONTRIBUTING.md>`_.

See our examples for walk-throughs of how to use the package.

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

**pywhy-graphs** is developed and maintained by pywhy.
To learn more about who specifically contributed to this codebase, see
`our contributors <https://github.com/py-why/pywhy-graphs/graphs/contributors>`_ page.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
